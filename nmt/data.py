src, trg = [], []

with open('nmt/data/news-commentary-v14.en-zh.tsv') as f:
    for i, line in enumerate(f):
        en, zh = '', ''
        try:
            en, zh = line.strip('\n').split('\t')
        except Exception as e:
            print(f'parse line <{i}> error.\n{e}')
            break
        if en != '' and zh != '':
            src.append(zh)
            trg.append(en)



