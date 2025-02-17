import multiprocessing
from wtpsplit import SaT

def main():
    # 1. 通常モデルの読み込み
    sat = SaT("sat-3l")
    # GPU利用（必要に応じて、TPUの場合は sat.to("xla:0") などを利用してください）
    sat.half().to("cuda")
    # 単一テキストの場合の分割結果を表示
    result = sat.split("This is a test This is another test.")
    print("sat.split(single text):", result)

    # 複数テキストの場合はイテレータが返るので、各テキストの結果を順次表示
    for sentences in sat.split(["This is a test This is another test.", "And some more texts..."]):
        print("sat.split(list):", sentences)

    # 2. '-sm' モデルの読み込み（一般的な文章分割タスク用）
    sat_sm = SaT("sat-3l-sm")
    sat_sm.half().to("cuda")
    result_sm = sat_sm.split("this is a test this is another test")
    print("sat_sm.split:", result_sm)

    # 3. LoRA で適応させたモデルの読み込み（言語・ドメイン適応用）
    sat_adapted = SaT("sat-3l", style_or_domain="ud", language="en")
    sat_adapted.half().to("cuda")
    result_adapted = sat_adapted.split("This is a test This is another test.")
    print("sat_adapted.split:", result_adapted)

if __name__ == '__main__':
    # Windows環境では freeze_support() の呼び出しが必要です
    multiprocessing.freeze_support()
    main()
