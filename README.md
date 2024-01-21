# Requirments
Pytorch==2.0.1

dgl==1.1.2

scikit-learn==1.3.0

python==3.9.18

# How to use
if your cuda is available, you can 
run ./code/main.py --gpu --enable_augmentation --enable_gumbel

else 
run ./code/main.py --enable_augmentation --enable_gumbel