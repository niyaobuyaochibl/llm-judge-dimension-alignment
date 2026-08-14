"""Run closed-source LLM judge validation through OpenRouter.

The script reads the upload-ready cases from gpt_upload_package/cases_all.jsonl
and calls OpenRouter's chat-completions API.  It never stores the API key;
provide it via the OPENROUTER_API_KEY environment variable.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path('/root/autodl-tmp/llm_diversity_eval')
OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'


def load_jsonl(path):
    rows = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path, obj):
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def existing_case_ids(path):
    done = set()
    if not path.exists():
        return done
    with path.open(encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                obj.get('status') == 'ok'
                and obj.get('case_id')
                and obj.get('verdict') in {'A', 'B', 'TIE'}
            ):
                done.add(obj['case_id'])
    return done


def parse_verdict(text):
    cleaned = (text or '').strip().upper()
    if cleaned in {'A', 'B', 'TIE'}:
        return cleaned
    m = re.search(r'\b(TIE|A|B)\b', cleaned)
    return m.group(1) if m else 'PARSE_ERROR'


def call_openrouter(
    api_key, model, system_prompt, user_prompt, max_tokens, temperature,
    timeout, retries, reasoning_effort=None, include_reasoning=False
):
    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        'temperature': temperature,
        'max_tokens': max_tokens,
    }
    if reasoning_effort:
        payload['reasoning'] = {'effort': reasoning_effort}
    if include_reasoning:
        payload['include_reasoning'] = True
    body = json.dumps(payload).encode('utf-8')
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        'HTTP-Referer': 'https://openai.com/codex',
        'X-Title': 'TKDE LLM Diversity Judge Validation',
    }
    last_error = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(OPENROUTER_URL, data=body, headers=headers, method='POST')
        try:
            start = time.time()
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode('utf-8')
            latency = time.time() - start
            data = json.loads(raw)
            content = data['choices'][0]['message']['content']
            return {
                'status': 'ok',
                'raw_output': content,
                'verdict': parse_verdict(content),
                'latency_sec': round(latency, 3),
                'usage': data.get('usage'),
                'response_model': data.get('model'),
            }
        except urllib.error.HTTPError as e:
            err_body = e.read().decode('utf-8', errors='replace')
            last_error = f'HTTP {e.code}: {err_body[:500]}'
            if e.code not in {408, 409, 425, 429, 500, 502, 503, 504}:
                break
        except Exception as e:
            last_error = repr(e)
        if attempt < retries:
            time.sleep(min(2 ** attempt, 20))
    return {'status': 'error', 'error': last_error, 'verdict': 'ERROR', 'raw_output': ''}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package-dir', default=str(PROJECT_ROOT / 'gpt_upload_package'))
    parser.add_argument('--out-dir', default=str(PROJECT_ROOT / 'results' / 'openrouter_validation'))
    parser.add_argument('--model', default='openai/gpt-4o')
    parser.add_argument('--limit', type=int, default=None, help='Run only the first N unfinished cases for smoke testing.')
    parser.add_argument('--max-tokens', type=int, default=8)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--timeout', type=int, default=90)
    parser.add_argument('--retries', type=int, default=3)
    parser.add_argument('--sleep', type=float, default=0.2)
    parser.add_argument('--api-key-file', default=None, help='Optional file containing the OpenRouter API key. Prefer /tmp and delete after use.')
    parser.add_argument('--reasoning-effort', default=None, help='Optional OpenRouter reasoning effort, e.g., none, low, medium, high.')
    parser.add_argument('--include-reasoning', action='store_true', help='Request reasoning content in responses when supported.')
    args = parser.parse_args()

    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key and args.api_key_file:
        api_key = Path(args.api_key_file).read_text(encoding='utf-8').strip()
    if not api_key:
        raise SystemExit('OPENROUTER_API_KEY is not set and --api-key-file was not provided.')

    package_dir = Path(args.package_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = load_jsonl(package_dir / 'cases_all.jsonl')
    safe_model = re.sub(r'[^A-Za-z0-9_.-]+', '_', args.model)
    raw_path = out_dir / f'raw_verdicts_{safe_model}.jsonl'
    verdict_path = out_dir / f'completed_{safe_model}.jsonl'
    done = existing_case_ids(raw_path)
    unfinished = [c for c in cases if c['case_id'] not in done]
    if args.limit is not None:
        unfinished = unfinished[:args.limit]

    print(f'Model: {args.model}', flush=True)
    print(f'Total cases: {len(cases)}; already done: {len(done)}; running now: {len(unfinished)}', flush=True)
    ok = errors = 0
    for idx, case in enumerate(unfinished, start=1):
        result = call_openrouter(
            api_key=api_key,
            model=args.model,
            system_prompt=case['system_prompt'],
            user_prompt=case['user_prompt'],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            retries=args.retries,
            reasoning_effort=args.reasoning_effort,
            include_reasoning=args.include_reasoning,
        )
        record = {
            'case_id': case['case_id'],
            'model': args.model,
            **result,
        }
        append_jsonl(raw_path, record)
        if result['status'] == 'ok' and result['verdict'] in {'A', 'B', 'TIE'}:
            append_jsonl(verdict_path, {'case_id': case['case_id'], 'verdict': result['verdict']})
            ok += 1
        else:
            errors += 1
        print(f"[{idx}/{len(unfinished)}] {case['case_id']} -> {record['verdict']} ({record['status']})", flush=True)
        if args.sleep:
            time.sleep(args.sleep)
    print(f'Done. ok={ok}, errors={errors}', flush=True)
    print(f'Raw: {raw_path}', flush=True)
    print(f'Verdicts: {verdict_path}', flush=True)


if __name__ == '__main__':
    main()
