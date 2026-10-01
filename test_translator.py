"""Offline regression checks for rewriting and the one-call/fallback contract.

Simulated responses test software behavior, not real model translation quality.
"""
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from translator import (BASE_DIR, Lexicon, OpenAIBackend, TranslationError, Translator,
                        load_configuration, save_gaps)


def entry(source, target, gloss=''):
    return {'source': source, 'target': target, 'gloss': gloss}


def match(text, entry_index, span_id=0, occurrence=0):
    return {'span_id': span_id, 'source_span': text, 'occurrence': occurrence,
            'entry_index': entry_index}


class FakeBackend:
    def __init__(self, response):
        self.response, self.calls = response, []

    def request(self, instructions, payload, schema, name):
        self.calls.append((instructions, copy.deepcopy(payload), schema, name))
        if isinstance(self.response, Exception):
            raise self.response
        return copy.deepcopy(self.response)


class TranslatorChecks(unittest.TestCase):
    def check_provenance(self, result):
        cursor = 0
        for item in result['provenance']:
            self.assertEqual(item['start'], cursor)
            self.assertEqual(result['source'][item['start']:item['end']], item['source'])
            cursor = item['end']
        self.assertEqual(cursor, len(result['source']))
        self.assertEqual(''.join(item['source'] for item in result['provenance']), result['source'])
        self.assertEqual(''.join(item['output'] for item in result['provenance']), result['han'])

    def test_longest_match_and_fully_local_output(self):
        lexicon = Lexicon([entry('什么', '勿'), entry('你', '汝'), entry('为什么', '做勿'), entry('去', '去')])
        backend = FakeBackend(RuntimeError('must never be called'))
        result = Translator(lexicon, backend).translate('你为什么去？')
        self.assertEqual(result['han'], '汝做勿去？')
        self.assertEqual(backend.calls, [])
        self.check_provenance(result)

    def test_identity_components_are_allowed_without_api(self):
        lexicon = Lexicon([entry('学', '学'), entry('校', '校')])
        backend = FakeBackend(RuntimeError('must never be called'))
        result = Translator(lexicon, backend).translate('学校')
        self.assertEqual(result['han'], '学校')
        self.assertEqual(result['statistics']['exact']['ratio'], 1)
        self.assertEqual(backend.calls, [])

    def test_unknown_content_survives_without_backend(self):
        result = Translator(Lexicon([entry('我', '我'), entry('去', '去')])).translate('我今天去学校学习计算机')
        self.assertEqual(result['han'], result['source'])
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['llm_status'], 'disabled')
        self.check_provenance(result)

    def test_replacements_are_not_matched_again(self):
        result = Translator(Lexicon([entry('你', '汝'), entry('汝', '伊')])).translate('你汝')
        self.assertEqual(result['han'], '汝伊')

    def test_whitespace_punctuation_and_unicode_preserved(self):
        source = '  你，\t今天\n𠊎e\u0301？  '
        result = Translator(Lexicon([entry('你', '汝')])).translate(source)
        self.assertEqual(result['han'], source.replace('你', '汝'))
        self.check_provenance(result)

    def test_empty_input_dictionary_and_separators_need_no_api(self):
        for lexicon, source in [(Lexicon([]), '任何新词？'), (Lexicon([entry('你', '汝')]), ''),
                                (Lexicon([entry('你', '汝')]), ' \n。！')]:
            with self.subTest(source=source):
                backend = FakeBackend(RuntimeError('must never be called'))
                result = Translator(lexicon, backend).translate(source)
                self.assertEqual(result['han'], source)
                self.assertEqual(backend.calls, [])
                self.check_provenance(result)

    def test_one_semantic_call_contains_context_locks_and_compact_dictionary(self):
        lexicon = Lexicon([entry('你', '汝', 'you'), entry('为什么', '做勿', 'why')])
        backend = FakeBackend({'matches': [match('怎么会', 1)]})
        result = Translator(lexicon, backend).translate('你怎么会今天才来学校？', ['在问迟到的原因。'])
        self.assertEqual(result['han'], '汝做勿今天才来学校？')
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(result['semantic_attempts'], 1)
        payload = backend.calls[0][1]
        self.assertEqual(payload['source'], result['source'])
        self.assertEqual(payload['context'], ['在问迟到的原因。'])
        self.assertEqual(payload['exact'], [{'start': 0, 'source': '你', 'output': '汝'}])
        self.assertEqual(payload['unresolved'][0]['text'], '怎么会今天才来学校')
        self.assertEqual(payload['dictionary'], lexicon.entries)
        self.assertTrue(all(set(item) == {'source', 'target', 'gloss'} for item in payload['dictionary']))
        self.check_provenance(result)

    def test_several_unresolved_spans_use_one_request(self):
        lexicon = Lexicon([entry('你', '汝'), entry('为什么', '做勿'), entry('他', '伊')])
        backend = FakeBackend({'matches': [match('咋', 1, 0), match('咋', 1, 1)]})
        result = Translator(lexicon, backend).translate('你咋来；他咋走')
        self.assertEqual(result['han'], '汝做勿来；伊做勿走')
        self.assertEqual(len(backend.calls), 1)
        self.check_provenance(result)

    def test_exact_lock_prevents_cross_boundary_rescue(self):
        lexicon = Lexicon([entry('你', '汝'), entry('会', '解', 'will'), entry('为什么', '做勿', 'why')])
        backend = FakeBackend({'matches': [match('怎么会', 2)]})
        result = Translator(lexicon, backend).translate('你怎么会今天才来')
        self.assertEqual(result['han'], '汝怎么解今天才来')
        self.assertEqual(result['provenance'][2]['method'], 'exact')
        self.assertEqual(len(backend.calls), 1)
        self.check_provenance(result)

    def test_api_error_invalid_json_and_empty_matches_all_preserve_local_result(self):
        for response in [TimeoutError('secret-sentinel'), RuntimeError('secret-sentinel'),
                         '{broken json', {'wrong_field': []}, {'matches': []}]:
            with self.subTest(response=type(response).__name__):
                backend = FakeBackend(response)
                result = Translator(Lexicon([entry('你', '汝')]), backend).translate('你今天学习计算机')
                self.assertEqual(result['han'], '汝今天学习计算机')
                self.assertEqual(result['status'], 'success')
                self.assertEqual(len(backend.calls), 1)
                self.assertNotIn('secret-sentinel', json.dumps(result))
                self.check_provenance(result)

    def test_invalid_matches_do_not_block_valid_independent_matches(self):
        lexicon = Lexicon([entry('你', '汝'), entry('为什么', '做勿')])
        patches = [match('咋', 999), match('不在原文', 1), match('咋', 1, span_id=-1),
                   match('咋', 1, occurrence=999999), match('', 1), match('咋', True),
                   {'translation': 'invented text'}, {**match('咋', 1), 'translation': 'invented text'},
                   match('咋', 1)]
        backend = FakeBackend({'matches': patches})
        result = Translator(lexicon, backend).translate('你咋迟到')
        self.assertEqual(result['han'], '汝做勿迟到')
        self.check_provenance(result)

    def test_repeated_expression_uses_requested_occurrence_only(self):
        lexicon = Lexicon([entry('为什么', '做勿')])
        backend = FakeBackend({'matches': [match('咋', 0, occurrence=1)]})
        result = Translator(lexicon, backend).translate('咋来咋走')
        self.assertEqual(result['han'], '咋来做勿走')
        self.check_provenance(result)

    def test_overlap_and_duplicate_patches_do_not_duplicate_output(self):
        lexicon = Lexicon([entry('为什么', '做勿')])
        backend = FakeBackend({'matches': [match('咋', 0), match('咋', 0), match('咋来', 0)]})
        result = Translator(lexicon, backend).translate('咋来')
        self.assertEqual(result['han'], '做勿来')
        self.check_provenance(result)

    def test_unsorted_matches_are_merged_in_source_order(self):
        lexicon = Lexicon([entry('为什么', '做勿'), entry('没有', '无')])
        backend = FakeBackend({'matches': [match('没', 1), match('咋', 0)]})
        result = Translator(lexicon, backend).translate('咋来没来')
        self.assertEqual(result['han'], '做勿来无来')
        self.check_provenance(result)

    def test_conflicting_source_is_unresolved_and_can_select_existing_target(self):
        lexicon = Lexicon([entry('在', '现今', 'progressive'), entry('在', '有', 'location'), entry('学', '学')])
        backend = FakeBackend({'matches': [match('在', 1)]})
        result = Translator(lexicon, backend).translate('在学')
        self.assertEqual(result['han'], '有学')
        self.assertEqual(result['provenance'][0]['method'], 'llm_semantic')
        self.assertEqual(Translator(lexicon).translate('在学')['han'], '在学')

    def test_same_source_same_target_can_still_match_locally(self):
        lexicon = Lexicon([entry('你', '汝', 'you'), entry('你', '汝', 'second person')])
        backend = FakeBackend(RuntimeError('must never be called'))
        self.assertEqual(Translator(lexicon, backend).translate('你')['han'], '汝')
        self.assertEqual(backend.calls, [])

    def test_provenance_statistics_use_source_characters_not_accuracy(self):
        lexicon = Lexicon([entry('你', '汝'), entry('为什么', '做勿')])
        backend = FakeBackend({'matches': [match('咋', 1)]})
        result = Translator(lexicon, backend).translate('你咋来？ ')
        stats = result['statistics']
        self.assertEqual(stats['total_characters'], 3)
        for method in ('exact', 'llm_semantic', 'passthrough'):
            self.assertEqual(stats[method]['characters'], 1)
            self.assertAlmostEqual(stats[method]['ratio'], 1 / 3)
        self.check_provenance(result)

    def test_passthrough_queue_ignores_punctuation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gaps.jsonl'
            translator = Translator(Lexicon([entry('你', '汝')]))
            save_gaps(translator.translate('你。'), path)
            self.assertFalse(path.exists())
            save_gaps(translator.translate('你今天？'), path)
            record = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual([item['source'] for item in record['passthrough']], ['今天'])


class DictionaryChecks(unittest.TestCase):
    def load_data(self, data):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'dictionary.json'
            path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
            return Lexicon.load(path)

    def test_simple_dictionary_can_grow_without_other_fields(self):
        data = [entry('为什么', '做勿', 'why'), entry('新词', '测试写法', 'test')]
        lexicon = self.load_data(data)
        self.assertEqual(Translator(lexicon).translate('新词')['han'], '测试写法')
        self.assertEqual(lexicon.entries, data)

    def test_old_dictionary_is_accepted_without_readings_or_ontology_validation(self):
        legacy = {'schema_version': 1, 'entries': [
            {'source_forms': ['学'], 'han': '学', 'sense': '学', 'kind': 'component'},
            {'source_forms': ['校'], 'han': '校', 'sense': '校', 'kind': 'component'},
            {'source_forms': ['旧词'], 'han': '忽略', 'status': 'disabled'},
        ]}
        lexicon = self.load_data(legacy)
        self.assertEqual(Translator(lexicon).translate('学校')['han'], '学校')
        self.assertEqual(len(lexicon.entries), 2)

    def test_empty_dictionary_is_valid(self):
        self.assertEqual(self.load_data([]).entries, [])

    def test_bad_dictionary_is_configuration_error(self):
        for data in [{}, [{'source': '', 'target': '汝', 'gloss': 'you'}], [entry('你', '')]]:
            with self.subTest(data=data), self.assertRaises(TranslationError):
                self.load_data(data)

    def test_original_recorded_characters_are_preserved_in_compact_dictionary(self):
        from migrate_dictionary import import_dictionary
        expected = import_dictionary(BASE_DIR / 'dictionary.tsv')
        self.assertEqual(Lexicon.load(BASE_DIR / 'lexicon.json').entries, expected)

    def test_import_without_readings_keeps_different_contextual_meanings(self):
        from migrate_dictionary import import_dictionary
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'words.tsv'
            path.write_text('词义\t文字\n在（表时间）\t现今\n在（方位）\t有\n你\t汝\n你\t汝\n', encoding='utf-8')
            result = import_dictionary(path)
        self.assertEqual(result, [entry('在', '现今', '在（表时间）'),
                                  entry('在', '有', '在（方位）'), entry('你', '汝', '你')])

    def test_import_cli_emits_simple_json_and_ignores_empty_readings(self):
        with tempfile.TemporaryDirectory() as directory:
            source, destination = Path(directory) / 'words.csv', Path(directory) / 'words.json'
            source.write_text('词义,文字,越南语声调拼音\n你,汝,\n', encoding='utf-8')
            process = subprocess.run([sys.executable, str(BASE_DIR / 'migrate_dictionary.py'),
                                      str(source), str(destination)], cwd=directory,
                                     capture_output=True, text=True, timeout=10)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(json.loads(destination.read_text(encoding='utf-8')), [entry('你', '汝', '你')])


class StartupChecks(unittest.TestCase):
    def test_cli_from_other_directory_returns_success_when_api_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            fake_sdk = Path(directory) / 'openai.py'
            fake_sdk.write_text('''class OpenAI:
    def __init__(self, **kwargs):
        assert kwargs["max_retries"] == 0
        self.responses = self
    def create(self, **kwargs):
        raise TimeoutError("secret-sentinel")
''', encoding='utf-8')
            environment = os.environ.copy()
            environment['PYTHONPATH'] = directory
            environment['OPENAI_API_KEY'] = 'test_config_value'
            process = subprocess.run([sys.executable, str(BASE_DIR / 'translator.py'), '你今天来？', '--json'],
                                     cwd=directory, env=environment, capture_output=True,
                                     text=True, timeout=10)
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout)
        self.assertEqual(result['han'], '汝今天来？')
        self.assertEqual(result['llm_status'], 'fallback')
        self.assertEqual(result['semantic_attempts'], 1)
        self.assertNotIn('secret-sentinel', process.stdout + process.stderr)


class BackendChecks(unittest.TestCase):
    def test_strict_small_response_and_no_sdk_retries(self):
        response = SimpleNamespace(status='completed', output=[], output_text='{"matches": []}')
        fake_client = SimpleNamespace(responses=SimpleNamespace(create=Mock(return_value=response)))
        constructor = Mock(return_value=fake_client)
        with patch.dict('sys.modules', {'openai': SimpleNamespace(OpenAI=constructor)}):
            backend = OpenAIBackend('gpt-5')
            self.assertFalse(constructor.called)
            result = Translator(Lexicon([entry('你', '汝')]), backend).translate('你今天')
        constructor.assert_called_once_with(timeout=30.0, max_retries=0)
        fake_client.responses.create.assert_called_once()
        options = fake_client.responses.create.call_args.kwargs
        self.assertTrue(options['text']['format']['strict'])
        self.assertEqual(options['max_output_tokens'], 2048)
        self.assertEqual(options['reasoning'], {'effort': 'minimal'})
        self.assertFalse(options['store'])
        self.assertEqual(result['han'], '汝今天')

    def test_incomplete_refusal_invalid_json_and_request_errors_fall_back(self):
        cases = [SimpleNamespace(status='incomplete', output=[], output_text='{}'),
                 SimpleNamespace(status='completed', output=[SimpleNamespace(content=[SimpleNamespace(type='refusal')])], output_text=''),
                 SimpleNamespace(status='completed', output=[], output_text='{bad json'),
                 RuntimeError('secret-sentinel')]
        for response in cases:
            with self.subTest(response=type(response).__name__):
                create = Mock(side_effect=response) if isinstance(response, Exception) else Mock(return_value=response)
                backend = OpenAIBackend('configured-model', SimpleNamespace(responses=SimpleNamespace(create=create)))
                result = Translator(Lexicon([entry('你', '汝')]), backend).translate('你今天')
                self.assertEqual(result['han'], '汝今天')
                self.assertEqual(result['llm_status'], 'fallback')
                self.assertNotIn('secret-sentinel', json.dumps(result))
                create.assert_called_once()
                self.assertNotIn('reasoning', create.call_args.kwargs)

    def test_missing_sdk_does_not_prevent_output(self):
        with patch.dict('sys.modules', {'openai': None}):
            backend = OpenAIBackend('gpt-5')
            local = Translator(Lexicon([entry('你', '汝')]), backend).translate('你')
            fallback = Translator(Lexicon([entry('你', '汝')]), backend).translate('你今天')
        self.assertEqual(local['han'], '汝')
        self.assertEqual(local['llm_status'], 'skipped')
        self.assertEqual(fallback['han'], '汝今天')
        self.assertEqual(fallback['llm_status'], 'fallback')


class ConfigurationChecks(unittest.TestCase):
    def test_configuration_loads_without_dotenv_and_accepts_quoted_values(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / 'config.env'
            path.write_text('OPENAI_API_KEY="test_config_value" # comment\nOPENAI_MODEL=gpt-5\n', encoding='utf-8')
            self.assertEqual(load_configuration(Path(directory)), path)
            self.assertEqual(os.environ['OPENAI_API_KEY'], 'test_config_value')
            self.assertEqual(os.environ['OPENAI_MODEL'], 'gpt-5')

    def test_configuration_preserves_existing_environment_and_env_file_priority(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'OPENAI_API_KEY': 'existing_test_value'}, clear=True):
            base = Path(directory)
            (base / '.env').write_text('OPENAI_API_KEY=file_test_value\nOPENAI_MODEL=env_file_model\n', encoding='utf-8')
            (base / 'config.env').write_text('OPENAI_MODEL=fallback_model\n', encoding='utf-8')
            self.assertEqual(load_configuration(base), base / '.env')
            self.assertEqual(os.environ['OPENAI_API_KEY'], 'existing_test_value')
            self.assertEqual(os.environ['OPENAI_MODEL'], 'env_file_model')

    def test_malformed_configuration_does_not_apply_partial_settings(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            base = Path(directory)
            (base / 'config.env').write_text('OPENAI_API_KEY=test_config_value\nOPENAI_MODEL="unfinished\n', encoding='utf-8')
            with self.assertRaises(TranslationError):
                load_configuration(base)
            self.assertNotIn('OPENAI_API_KEY', os.environ)


if __name__ == '__main__':
    unittest.main()
