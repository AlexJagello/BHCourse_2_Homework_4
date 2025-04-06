import torch.nn as nn
import torch.nn.functional as F
import torch

def saveModel(root, generator, discriminator, loses, s_loses):
    torch.save(generator.state_dict(), root +"generator.pt")
    torch.save(discriminator.state_dict(), root + "discriminator.pt")

    from matplotlib import pyplot as plt
    plt.plot([sublist[0] for sublist in loses], label="Generator Loss")
    plt.plot([sublist[1] for sublist in loses], label="Discriminator Loss")
    plt.legend()
    plt.title("loses")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(root + 'loses.png')
    plt.clf()

    from matplotlib import pyplot as plt
    plt.plot([sublist[0] for sublist in s_loses], label="Generator Loss")
    plt.plot([sublist[1] for sublist in s_loses], label="Discriminator Loss")
    plt.legend()
    plt.title("sample loses")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(root + 's_loses.png')
    plt.show()


    