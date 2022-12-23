source activate mlbd

# vgg16

# python -u run.py --lr 1e-4 --initmodel True --loadwt False --weightsroot ./weights --weights None --model xvgg16 --saveweights model-base.pth --dataset base.npz --epoch 60
# python -u run.py --lr 1e-4 --initmodel False --loadwt True --weightsroot ./weights --weights model-base.pth --model xvgg16 --saveweights model-double.pth --dataset double.npz --epoch 60
# python -u run.py --lr 1e-4 --initmodel False --loadwt True --weightsroot ./weights --weights model-double.pth --model xvgg16 --saveweights model-bd.pth --dataset bd.npz --epoch 50
# python -u run.py --lr 1e-4 --initmodel False --loadwt True --weightsroot ./weights --weights model-bd.pth --model xvgg16 --saveweights model-multi.pth --dataset multi.npz --epoch 60
# python -u run.py --lr 1e-4 --initmodel False --loadwt True --weightsroot ./weights --weights model-multi.pth --model xvgg16 --saveweights model-full.pth --dataset full.npz --epoch 70
# python -u run.py --lr 1e-3 --initmodel False --loadwt True --weightsroot ./weights --weights model-full.pth --model xvgg16 --saveweights model-full-sgd.pth --dataset full.npz --epoch 30 --optim sgd

# python -u run.py --target eval --initmodel False --loadwt True --weightsroot ./weights --weights model-full-sgd.pth --model xvgg16

# resnet50

# python -u run.py --lr 1e-4 --initmodel True --loadwt False --weightsroot ./weights --weights None --model xresnet50 --saveweights model-base.pth --dataset base.npz --epoch 20
# python -u run.py --lr 1e-4 --initmodel False --loadwt True --weightsroot ./weights --weights model-base.pth --model xresnet50 --saveweights model-double.pth --dataset double.npz --epoch 20
# python -u run.py --lr 1e-4 --initmodel False --loadwt True --weightsroot ./weights --weights model-double.pth --model xresnet50 --saveweights model-bd.pth --dataset bd.npz --epoch 30
# python -u run.py --lr 1e-4 --initmodel False --loadwt True --weightsroot ./weights --weights model-bd.pth --model xresnet50 --saveweights model-multi.pth --dataset multi.npz --epoch 30
# python -u run.py --lr 1e-4 --initmodel False --loadwt True --weightsroot ./weights --weights model-multi.pth --model xresnet50 --saveweights model-full.pth --dataset full.npz --epoch 70
# python -u run.py --lr 1e-3 --initmodel False --loadwt True --weightsroot ./weights --weights model-full.pth --model xresnet50 --saveweights model-full-sgd.pth --dataset full.npz --epoch 30 --optim sgd

# python -u run.py --target eval --initmodel False --loadwt True --weightsroot ./weights --weights model-full-sgd.pth --model xresnet50

# transformer

# python -u run.py --lr 1e-3 --initmodel True --loadwt False --weightsroot ./weights --weights None --model xvit --saveweights model-base.pth --dataset base.npz --epoch 20
# python -u run.py --lr 1e-3 --initmodel False --loadwt True --weightsroot ./weights --weights model-base.pth --model xvit --saveweights model-double.pth --dataset double.npz --epoch 20
# python -u run.py --lr 1e-3 --initmodel False --loadwt True --weightsroot ./weights --weights model-double.pth --model xvit --saveweights model-bd.pth --dataset bd.npz --epoch 10
# python -u run.py --lr 1e-3 --initmodel False --loadwt True --weightsroot ./weights --weights model-bd.pth --model xvit --saveweights model-multi.pth --dataset multi.npz --epoch 20
# python -u run.py --lr 1e-3 --initmodel False --loadwt True --weightsroot ./weights --weights model-multi.pth --model xvit --saveweights model-full.pth --dataset full.npz --epoch 200
# python -u run.py --lr 1e-3 --initmodel False --loadwt True --weightsroot ./weights --weights model-full.pth --model xvit --saveweights model-full-sgd.pth --dataset full.npz --epoch 30 --optim sgd

# python -u run.py --target eval --initmodel False --loadwt True --weightsroot ./weights --weights model-full-sgd.pth --model xvit

# neck

# python -u run.py --lr 1e-3 --initmodel True --loadwt False --weightsroot ./weights --weights None --model neck --saveweights model-full-stage1.pth --dataset full.npz --epoch 5 --bts 128
# python -u run.py --lr 1e-4 --initmodel False --loadwt True --weightsroot ./weights --weights model-full-stage1.pth --model neck --saveweights model-full-stage2.pth --dataset full.npz --epoch 25 --bts 128
# python -u run.py --lr 1e-5 --initmodel False --loadwt True --weightsroot ./weights --weights model-full-stage2.pth --model neck --saveweights model-full-stage3.pth --dataset full.npz --epoch 25 --bts 128
# python -u run.py --lr 1e-3 --initmodel False --loadwt True --weightsroot ./weights --weights model-full-stage3.pth --model neck --saveweights model-full-sgd.pth --dataset full.npz --epoch 50 --optim sgd --bts 128
# python -u run.py --lr 1e-4 --initmodel False --loadwt True --weightsroot ./weights --weights model-full-sgd.pth --model neck --saveweights model-full-final.pth --dataset full.npz --epoch 30 --optim sgd --bts 128 --final True
# python -u run.py --lr 1e-4 --initmodel False --loadwt True --weightsroot ./weights --weights model-full-final.pth --model neck --saveweights model-full-final-2.pth --dataset full.npz --epoch 20 --optim sgd --bts 64 --final True

# python -u run.py --target eval --initmodel False --loadwt True --weightsroot ./weights --weights model-full-final-2.pth --model neck

# python -u run.py --target eval --stage final-test --initmodel False --loadwt True --weightsroot ./weights --weights model-full-final-2.pth --model neck --result ./data/result/Group5.csv