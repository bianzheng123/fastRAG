import os

os.system('python3 scripts/indexing/create_plaid.py            \
         --username zhengbian                      \
         --dataset wikipedia                    \
         --doc-max-length 180                      \
         --query-max-length 60                      \
         --kmeans-iterations 20                     \
         --gpu 1 --ranks 1')

os.system('proxychains python3 self/plaid_retrieval.py')
