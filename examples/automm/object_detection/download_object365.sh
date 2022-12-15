# Download images
url='https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/train/zhiyuan_objv2_train.tar.gz'
wget $url
url='https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/train/'
for((f=0;f<=50;f++)); do
  echo 'Downloading' $url'patch'$f'.tar.gz' '...'
  wget $url'patch'$f'.tar.gz'
done
wait # finish background tasks

# Download images
url='https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/val/zhiyuan_objv2_val.json'
wget $url
url='https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/val/images/v1/'
for((f=0;f<=15;f++)); do
  echo 'Downloading' $url'patch'$f'.tar.gz' '...'
  wget $url'patch'$f'.tar.gz'
done
url='https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/val/images/v2/'
for((f=16;f<=43;f++)); do
  echo 'Downloading' $url'patch'$f'.tar.gz' '...'
  wget $url'patch'$f'.tar.gz'
done
wait # finish background tasks

