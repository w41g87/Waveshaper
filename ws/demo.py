import ws_only as ws
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

def img_optimization(input, nodes, pump, width, strides=1, **kwargs):

    target = np.array(ImageOps.grayscale(Image.open(input).resize((nodes, nodes), resample=Image.LANCZOS)))
    target = target / np.max(target)

    pred, loss, output = ws.jsi_backprop(nodes=nodes, npump=pump, width=width, strides=strides, target=target, EPOCHS=1e3 if kwargs['epochs'] is None else kwargs['epochs'])
    print('loss: ', loss[-1])
    print('params: \n', pred)
    try:
        plt.close(fig)
        plt.close(fig2)
    except:
        pass
    fig = ws.pltCtst(target, output, 0, 0, nodes, nodes)
    if kwargs['output'] is not None:
        plt.savefig(kwargs['output'])
    fig2, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), dpi=50)
    ax.plot(range(len(loss)), loss, label='loss')
    ax.legend(loc='best')
    ax.grid()
    
    if pred.get('int', False):
        raise KeyboardInterrupt()
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Import and image and optimize the waveshaper parameters')
    parser.add_argument('-i', '--input', metavar='path', required=True, type=str, 
                        help='the path to the image')
    parser.add_argument('-o', '--output', metavar='path', required=False, type=str, 
                        help='the path to the output image')
    parser.add_argument('-e', '--epochs', metavar='path', required=False, type=float, 
                        help='number of epochs to run the optimization')
    parser.add_argument('-n', '--nodes', metavar='nodes', required=True, type=int, 
                        help='total output dimension')
    parser.add_argument('-p', '--pump', metavar='nodes', required=True, type=int, 
                        help='number of pump nodes')
    parser.add_argument('-w', '--width', metavar='nodes', required=True, type=int, 
                        help='number of spdc nodes')
    parser.add_argument('-s', '--strides', metavar='strides', required=False, type=int, 
                        help='macropixel dimension')
    args = parser.parse_args()
    img_optimization(**vars(args))