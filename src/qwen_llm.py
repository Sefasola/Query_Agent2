import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def _has_accelerate() -> bool:
    try:
        import accelerate  # noqa: F401
        return True
    except Exception:
        return False

def _try_4bit():
    try:
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    except Exception:
        return None

class QwenChat:
    def __init__(self, model_name="Qwen/Qwen3-4B-Instruct-2507", temperature=0.0, max_new_tokens=96):
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        has_cuda = torch.cuda.is_available()
        has_acc = _has_accelerate()
        bnb = _try_4bit() if has_cuda and has_acc else None  # 4-bit için accelerate şart

        kwargs = {}
        if bnb is not None:
            # 4-bit yükleme (accelerate gerekir)
            kwargs["quantization_config"] = bnb
            kwargs["device_map"] = "auto"
        elif has_acc and has_cuda:
            # GPU var ve accelerate var → otomatik yerleşim
            kwargs["device_map"] = "auto"
            kwargs["torch_dtype"] = torch.bfloat16
        else:
            # accelerate yoksa device_map verme; klasik .to(device) yaparız
            kwargs["torch_dtype"] = torch.bfloat16 if has_cuda else torch.float32

        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

        if not kwargs.get("device_map"):
            # accelerate kullanmıyorsak elle cihaza taşı
            self.model.to("cuda" if has_cuda else "cpu")

        self.eos = self.tok.eos_token_id

    def chat(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        tpl = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok([tpl], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=False,
                eos_token_id=self.eos,
            )
        text = self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text.strip()
