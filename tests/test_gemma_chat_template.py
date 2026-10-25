from jinja2.sandbox import ImmutableSandboxedEnvironment

from agrimanager.adapter.trainer.main_ppo import GEMMA_SYSTEM_CHAT_TEMPLATE
from verl.utils.chat_template import initialize_system_prompt


class FakeGemmaTokenizer:
    def __init__(self):
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        self._template = env.from_string(GEMMA_SYSTEM_CHAT_TEMPLATE)

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True, **kwargs):
        rendered = self._template.render(
            messages=messages,
            bos_token="<bos>",
            add_generation_prompt=add_generation_prompt,
            raise_exception=lambda message: (_ for _ in ()).throw(RuntimeError(message)),
            **kwargs,
        )
        if not tokenize:
            return rendered
        return list(rendered.encode("utf-8"))


def test_gemma_template_merges_initial_system_message():
    tokenizer = FakeGemmaTokenizer()

    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Follow crop-management policy."},
            {"role": "user", "content": "State the next action."},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )

    assert "<start_of_turn>user\nFollow crop-management policy.\n\nState the next action." in rendered
    assert rendered.endswith("<start_of_turn>model\n")


def test_gemma_template_supports_verl_system_prompt_probe():
    tokenizer = FakeGemmaTokenizer()

    system_prompt = initialize_system_prompt(tokenizer)

    assert isinstance(system_prompt, list)
