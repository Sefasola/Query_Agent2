import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def _try_load_4bit():
    try:
        from bitsandbytes import __version__ as _  # noqa
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    except Exception:
        return None

class QwenClient:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B-Instruct",
        temperature: float = 0.0,
        max_new_tokens: int = 96,
        system_prompt: str | None = None,
    ):
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt or ""

        quant = _try_load_4bit()
        kwargs = {"device_map": "auto"}
        if quant:
            kwargs["quantization_config"] = quant
        else:
            kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.eos = self.tokenizer.eos_token_id

    def generate(self, page_text: str, question: str) -> str:
        user = (
            f"Soru: {question}\n\n"
            f"Sayfa metni:\n\"\"\"\n{page_text}\n\"\"\"\n\n"
            "Çıktı biçimi:\n- Varsa: <<<tek satır tam kopya>>>\n- Yoksa: BELİRTİLMEMİŞ"
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user},
        ]
        tpl = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([tpl], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=False,
                eos_token_id=self.eos,
            )
        text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text.strip()
