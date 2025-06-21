from transformers import AutoTokenizer, BertForSequenceClassification, BertConfig
import torch

def load_classification_model(device: str):
    """Загружаем обученные модель классификации и токенизатор"""
    try:
        path_to_cls_model = "models/classification_model"

        cls_config = BertConfig.from_pretrained(
            path_to_cls_model,
            num_labels=3,
            hidden_dropout_prob=0.1,
        )
        cls_model = BertForSequenceClassification.from_pretrained(
            path_to_cls_model,
            config=cls_config
        )
        cls_tokenizer = AutoTokenizer.from_pretrained(path_to_cls_model)

        cls_model.to(device)

        return cls_model, cls_tokenizer
    except Exception as e:
        print(f"Error - {e}")
        return

    

def perform_input_classification(text: str, model: BertForSequenceClassification, 
                                 tokenizer: AutoTokenizer, device: str) -> int:
    """Производит классификацию ответа клиента.
    
    Возвращает int принадлежащий {0, 1, 2}, где 
    0 - нейтральный ответ, чат бот продолжает узнавать больше о клиенте
    1 - упоминание собаки, чат бот предлагает купить ошейник
    2 - упоминание отсутствия денег, чат бот навязывает участие в MLM
    """
    try:
        output = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )

        input_ids = output["input_ids"].to(device)
        attention_mask = output["attention_mask"].to(device)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )[0]
        label = torch.argmax(logits, dim=1).cpu().numpy().tolist()[0]
        print(f"Label: {label}, type {type(label)}")
        return label
    except Exception as e:
        print(f"Error - {e}")