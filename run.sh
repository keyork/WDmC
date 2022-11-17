source activate mlbd

python -u run.py --lr 1e-4 --initmodel True --loadwt False --weights None --saveweights ./weights/model0.pth --dataset base.npz --epoch 50
python -u run.py --lr 1e-5 --initmodel False --loadwt True --weights ./weights/model0.pth --saveweights ./weights/model1.pth --dataset base.npz --epoch 50
python -u run.py --lr 1e-6 --initmodel False --loadwt True --weights ./weights/model1.pth --saveweights ./weights/model2.pth --dataset base.npz --epoch 20

python -u run.py --lr 1e-4 --initmodel False --loadwt True --weights ./weights/model2.pth --saveweights ./weights/model3.pth --dataset double.npz --epoch 20
python -u run.py --lr 1e-5 --initmodel False --loadwt True --weights ./weights/model3.pth --saveweights ./weights/model4.pth --dataset double.npz --epoch 50
python -u run.py --lr 1e-6 --initmodel False --loadwt True --weights ./weights/model4.pth --saveweights ./weights/model5.pth --dataset double.npz --epoch 20

python -u run.py --lr 1e-4 --initmodel False --loadwt True --weights ./weights/model5.pth --saveweights ./weights/model6.pth --dataset bd.npz --epoch 20
python -u run.py --lr 1e-5 --initmodel False --loadwt True --weights ./weights/model6.pth --saveweights ./weights/model7.pth --dataset bd.npz --epoch 20

python -u run.py --lr 1e-4 --initmodel False --loadwt True --weights ./weights/model7.pth --saveweights ./weights/model8.pth --dataset multi.npz --epoch 20
python -u run.py --lr 1e-5 --initmodel False --loadwt True --weights ./weights/model8.pth --saveweights ./weights/model9.pth --dataset multi.npz --epoch 20

python -u run.py --lr 1e-4 --initmodel False --loadwt True --weights ./weights/model9.pth --saveweights ./weights/model10.pth --dataset full.npz --epoch 30
python -u run.py --lr 1e-4 --initmodel False --loadwt True --weights ./weights/model10.pth --saveweights ./weights/model11.pth --dataset full.npz --epoch 10
python -u run.py --lr 1e-5 --initmodel False --loadwt True --weights ./weights/model11.pth --saveweights ./weights/model12.pth --dataset full.npz --epoch 50

python -u run.py --lr 1e-3 --initmodel False --loadwt True --weights ./weights/model12.pth --saveweights ./weights/model13.pth --dataset full.npz --epoch 30 --optim sgd

python -u run.py --target eval --initmodel False --loadwt True --weights ./weights/model13.pth