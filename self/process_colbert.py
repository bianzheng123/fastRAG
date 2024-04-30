def process_colbert(username: str, dataset: str):
    dataset_filename = f'/home/{username}/Dataset/vector-set-similarity-search/RawData/{dataset}/document/collection.tsv'
    data_l = []
    with open(dataset_filename, 'r') as f:
        for line_idx, line in enumerate(f):
            if line_idx % (1000 * 1000) == 0:
                print(f'{line_idx // 1000 // 1000}M', end=' ', flush=True)

            pid, passage, *rest = line.strip('\n\r ').split('\t')
            data_l.append((line_idx, passage))

    proc_collection_filename = f'/home/{username}/fastRAG/data/collection/{dataset}-collection.tsv'
    with open(proc_collection_filename, 'w') as f:
        for line_idx, passage in data_l:
            f.write(f'{line_idx}\t{passage}\n')
    print("process finish")


if __name__ == '__main__':
    username = 'zhengbian'
    dataset = 'lotte-lifestyle'
    process_colbert(username=username, dataset=dataset)
