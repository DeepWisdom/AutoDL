#!/usr/bin/python
# Author: Liu Zhengying
# Date: 10 Apr 2019
# Description: This script downloads the public datasets used in AutoDL challenges and
#   put them under the folder AutoDL_public_data/. This script supports
#   breakpoint resume, which means that you can recover downloading from where
#   your network broke down.
# Usage:
#   python download_public_datasets.py

import os
import sys

def main(*argv):
  dataset_names = ['Munster', 'City', 'Chucky','Pedro', 'Decal', 'Hammer', # images
                   'Kreatur', 'Kreatur3', 'Katze', 'Kraut',                # videos
                   'data01', 'data02', 'data03', 'data04', 'data05',       # speech
                   'O1', 'O2', 'O3', 'O4', 'O5',                           # text
                   'Adult', 'Dilbert', 'Digits', 'Madeline']               # tabular
  dataset_names = [ # images
                                  # videos
                   'data04', 'data05',       # speech
                   #'O1', 'O2', 'O3', 'O4', 'O5',                           # text
                   #'Adult', 'Dilbert', 'Digits', 'Madeline']               # tabular
                   ]

  dataset_names = ['Munster', 'City', 'Chucky','Pedro', 'Decal', 'Hammer', # images
                   'Kreatur', 'Kreatur3', 'Katze', 'Kraut',]                # videos

  data_urls = {
      'Munster':'https://autodl.lri.fr/my/datasets/download/d29000a6-b5b8-4ccf-9050-2686af874a71',
      'City':'https://autodl.lri.fr/my/datasets/download/cf0f810e-4818-4c8a-bf48-cbf9b6599928',
      'Chucky':'https://autodl.lri.fr/my/datasets/download/cf2176c2-5454-4d07-9c4e-758e3c5bcb31',
      'Pedro':'https://autodl.lri.fr/my/datasets/download/d556ca67-01c7-4a8d-9a74-a2bd9c06414d',
      'Decal':'https://autodl.lri.fr/my/datasets/download/31a34c03-a75c-4e0f-b72d-3723ba303dac',
      'Hammer':'https://autodl.lri.fr/my/datasets/download/3507841e-59fe-4598-a27e-a9e170b26e44',
      'Kreatur':'https://autodl.lri.fr/my/datasets/download/c240df57-b144-41df-a059-05bc859d2621',
      'Kreatur3':'https://autodl.lri.fr/my/datasets/download/08c2afcd-74b1-4c5e-8b93-9f6c9a96add2',
      'Kraut':'https://autodl.lri.fr/my/datasets/download/a1d9f706-cf8d-4a63-a544-552d6b85d6c4',
      'Katze':'https://autodl.lri.fr/my/datasets/download/611a42fa-da42-4a30-8c7a-69230d9eeb92',
      'data01':'https://autodl.lri.fr/my/datasets/download/c15f1b70-4f07-4e9e-9817-d785b1674966',
      'data02':'https://autodl.lri.fr/my/datasets/download/3961f962-88db-47ee-a756-2152753ba900',
      'data03':'https://autodl.lri.fr/my/datasets/download/a97ddb39-1470-4a80-81b0-1c26dfa29335',
      'data04':'https://autodl.lri.fr/my/datasets/download/c8d15be0-e1fa-4899-940e-0d7e1794a835',
      'data05':'https://autodl.lri.fr/my/datasets/download/cccc9147-7d1f-4119-888b-1b87f142b721',
      'O1':'https://autodl.lri.fr/my/datasets/download/4b98c65f-1922-4ff4-8e2a-ab7a022ef1da',
      'O2':'https://autodl.lri.fr/my/datasets/download/f831b0d6-0a53-4c93-b9cf-8cf1f2128d24',
      'O3':'https://autodl.lri.fr/my/datasets/download/4545d366-12f4-442c-87e4-f908fcd79698',
      'O4':'https://autodl.lri.fr/my/datasets/download/2bdf5e4e-8d02-4c85-98b2-0b28a6176db9',
      'O5':'https://autodl.lri.fr/my/datasets/download/09ec4daf-fba2-41e1-80d0-429772d59d58',
      'Adult':'https://autodl.lri.fr/my/datasets/download/4ad27a85-4932-409b-a33d-a3b1c4ec1893',
      'Dilbert':'https://autodl.lri.fr/my/datasets/download/71f517b0-85c2-4a7d-8df3-d2a5998a9d78',
      'Digits':'https://autodl.lri.fr/my/datasets/download/03e69995-2b8b-4f60-b43b-4458aa51e9c0',
      'Madeline':'https://autodl.lri.fr/my/datasets/download/1d7910ca-ee43-41fc-aca9-0dfcd800d93b'
  }
  solution_urls = {
      'Munster':'https://autodl.lri.fr/my/datasets/download/5a24d8f3-dfb6-4935-b798-14baccda695f',
      'City':'https://autodl.lri.fr/my/datasets/download/c64e3ebb-664f-45f1-8666-1054d262a85c',
      'Chucky':'https://autodl.lri.fr/my/datasets/download/ba4837bf-275d-43a6-a481-d03dce7ba127',
      'Pedro':'https://autodl.lri.fr/my/datasets/download/9993ea27-955e-4faa-9d28-4a7dfe1fcc55',
      'Decal':'https://autodl.lri.fr/my/datasets/download/cc93c74c-2732-4e7d-ae7f-a2c3bc555360',
      'Hammer':'https://autodl.lri.fr/my/datasets/download/e5b6188f-a377-4a5d-bbe1-a586716af487',
      'Kraut':'https://autodl.lri.fr/my/datasets/download/47ff016d-cc66-47a9-945d-bc01fd9096c9',
      'Katze':'https://autodl.lri.fr/my/datasets/download/a04de92e-b04b-49a6-96c2-5910c64f9b3c',
      'Kreatur':'https://autodl.lri.fr/my/datasets/download/31ecdb19-c25a-420f-9764-8d1783705deb',
      'Kreatur3':'https://autodl.lri.fr/my/datasets/download/10e04890-f05a-4004-a499-1cc167769edd',
      'data01':'https://autodl.lri.fr/my/datasets/download/358a227e-986d-48ad-a994-70b12a9bfcc3',
      'data02':'https://autodl.lri.fr/my/datasets/download/9be871dc-1356-4600-962e-9a43154a1e38',
      'data03':'https://autodl.lri.fr/my/datasets/download/a6baf245-03a6-42b5-b870-df57e3a27723',
      'data04':'https://autodl.lri.fr/my/datasets/download/32fd9ca3-865c-4135-a70c-1543160cf6ab',
      'data05':'https://autodl.lri.fr/my/datasets/download/4f802113-a899-40a8-b293-e119dd7c54f5',
      'O1':'https://autodl.lri.fr/my/datasets/download/888e1c1c-a39c-40b9-b6f7-cc2eff7a299d',
      'O2':'https://autodl.lri.fr/my/datasets/download/b6513cb2-e4f2-46a8-a7cf-1d8441a00a56',
      'O3':'https://autodl.lri.fr/my/datasets/download/4bb6ebd3-f991-4a6b-8cd1-864a0f3a1abd',
      'O4':'https://autodl.lri.fr/my/datasets/download/e09837eb-8144-4850-ad38-c1ba81426c0b',
      'O5':'https://autodl.lri.fr/my/datasets/download/18c8e3eb-2341-41ed-9a44-8d2c94042c30',
      'Adult':'https://autodl.lri.fr/my/datasets/download/c125d32c-3e89-456a-a82d-760fc4b60e4c',
      'Dilbert':'https://autodl.lri.fr/my/datasets/download/7734cc00-1583-44c8-80f5-156a11b12952',
      'Digits':'https://autodl.lri.fr/my/datasets/download/e29c6cb2-8748-4e26-9c91-66a2b0dd41c2',
      'Madeline':'https://autodl.lri.fr/my/datasets/download/a86e0e7f-9b07-44f1-92ba-0a5f72cddb6b'
  }

  def _HERE(*args):
      h = os.path.dirname(os.path.realpath(__file__))
      return os.path.abspath(os.path.join(h, *args))
  starting_kit_dir = _HERE()
  public_date_dir = os.path.join(starting_kit_dir, 'AutoDL_public_data')

  for dataset_name in dataset_names:
    msg = "Downloading data files and solution file for the dataset {}..."\
          .format(dataset_name)
    le = len(msg)
    print('\n' + '#'*(le+10))
    print('#'*4+' ' + msg + ' '+'#'*4)
    print('#'*(le+10) + '\n')
    data_url = data_urls[dataset_name]
    solution_url = solution_urls[dataset_name]
    dataset_dir = os.path.join(public_date_dir, dataset_name)
    os.system('mkdir -p {}'.format(dataset_dir))
    data_zip_file = os.path.join(dataset_dir, dataset_name + '.data.zip')
    solution_zip_file = os.path.join(dataset_dir,
                                     dataset_name + '.solution.zip')
    os.system('wget -q --show-progress -c -N {} -O {}'\
              .format(data_url, data_zip_file))
    os.system('wget -q --show-progress -c -N {} -O {}'\
              .format(solution_url, solution_zip_file))
    os.system('unzip -n -d {} {}'\
              .format(dataset_dir, data_zip_file))
    os.system('unzip -n -d {} {}'\
              .format(dataset_dir, solution_zip_file))
  print("\nFinished downloading {} public datasets: {}"\
        .format(len(dataset_names),dataset_names))
  print("Now you should find them under the directory: {}"\
        .format(public_date_dir))

if __name__ == '__main__':
  main(sys.argv)
