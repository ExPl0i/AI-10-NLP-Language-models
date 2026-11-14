"""Подготовка к занятию «Языковые модели» (запускать на компьютере с интернетом).

Сохраняет модель cointegrated/rubert-tiny2 (около 120 МБ) в папку models/rubert-tiny2,
чтобы задания Г3–Г5 работали в аудитории без интернета:
    python prepare_data.py
Ноутбук сам найдет папку models/rubert-tiny2 рядом с собой. Корпус, правила BPE,
готовая контрольная точка GPT для легкого режима и фразы водителя уже лежат в папке data.
"""
from pathlib import Path

BERT_NAME = 'cointegrated/rubert-tiny2'

if __name__ == '__main__':
    from transformers import AutoModel, AutoTokenizer
    target = Path(__file__).parent / 'models' / 'rubert-tiny2'
    if (target / 'config.json').exists():
        print('уже есть:', target)
    else:
        print('скачиваю', BERT_NAME)
        AutoTokenizer.from_pretrained(BERT_NAME).save_pretrained(target)
        AutoModel.from_pretrained(BERT_NAME).save_pretrained(target)
        size = sum(f.stat().st_size for f in target.rglob('*') if f.is_file())
        print(f'готово: {target} ({size / 2**20:.0f} МБ)')
