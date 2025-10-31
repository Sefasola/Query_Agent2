from langchain_community.embeddings import HuggingFaceEmbeddings

def build_embeddings(model_name: str = "intfloat/multilingual-e5-base"):
    # E5 için 'instruction' prefix gerekmiyor; LC bunu içermiyor.
    return HuggingFaceEmbeddings(model_name=model_name, encode_kwargs={"normalize_embeddings": True})
