import numpy as np
import torch
from visdom import Visdom


class Scalar(object):
    def __init__(self, opts=dict()):
        self.opts = opts
        self.opts.setdefault('showlegend', True)
        self.opts.setdefault('xlabel', 'epoch')

    def _rectify(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach()
            if x.is_cuda:
                x = x.cpu()
            x = x.numpy()

        if isinstance(x, np.ndarray):
            x = x.squeeze()
            if x.ndim == 0:
                x = x[np.newaxis]
        else:
            x = np.array([x])

        return x

    def update(self, vis, win, name, x, y, remove=False, opts=dict()):
        x, y = self._rectify(x), self._rectify(y)

        assert len(x) == len(y)

        _opts = self.opts.copy()
        _opts.update(opts)

        if remove:
            vis.line(X=x, Y=y, win=win, name=name, update='remove')

        res = vis.line(X=x, Y=y, win=win, name=name,
                       update='append',
                       opts=_opts)
        if res == 'win does not exist':
            vis.line(X=x, Y=y, win=win,
                     name=name, opts=_opts)


class Image2D(object):
    def __init__(self):
        pass

    def _rectify(self, image):
        assert isinstance(image, torch.Tensor)
        image = image.detach()
        if image.is_cuda:
            image = image.cpu()

        if image.dim() == 3:
            image = image.unsqueeze(0)
        assert image.dim() == 4
        assert image.size(1) == 1 or image.size(1) == 3

        return image

    def update(self, vis, win, name, images, caption=None, nrow=3):
        images = self._rectify(images)

        caption = caption if caption is not None else name
        vis.images(images, win=win, nrow=nrow,
                   opts=dict(title=name, caption=caption))


class Image3D(object):
    '''TODO indexing flavor'''

    def __init__(self):
        pass

    def _rectify(self, image):
        assert isinstance(image, torch.Tensor)
        image = image.detach()
        if image.is_cuda:
            image = image.cpu()

        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.dim() == 4:
            image = image.unsqueeze(0)
        assert image.dim() == 5
        assert image.size(1) == 1 or image.size(1) == 3

        return image

    def _same_size(self, images):
        shapes = [image.shape for image in images]
        pannel_shape = np.max(shapes, axis=0)
        pannels = [torch.ones(tuple(pannel_shape)) for _ in range(len(images))]
        for pannel, image in zip(pannels, images):
            s = image.shape
            pannel[:, :s[1], :s[2]] *= image

        return pannels

    def _slice_stack(self, image, nim):
        if nim == 1:
            ci = (np.array(image.shape[1:])//2)
            images = [image[:, ci[0], :, :],
                      image[:, :, ci[1], :],
                      image[:, :, :, ci[2]]]
        else:
            getters = [np.linspace(0, axis-1, nim, dtype=int).tolist()
                       for axis in image.shape[1:]]
            images = []
            for i0, i1, i2 in zip(*getters):
                images += [image[:, i0, :, :]]
                images += [image[:, :, i1, :]]
                images += [image[:, :, :, i2]]

        images = torch.stack(self._same_size(images))
        return images

    def update(self, vis, win, name, images):
        images = self._rectify(images)

        images = torch.cat([self._slice_stack(images[i, ...], nim=1)
                            for i in range(images.shape[0])])

        vis.images(images, win=win, nrow=3,
                   opts=dict(caption=name, title=name))


class VisdomSummary(object):
    def __init__(self, port=None, env=None):
        self.vis = Visdom(port=port, env=env)
        self.opts = dict()

    def scalar(self, win, name, x, y, remove=False):
        if not hasattr(self, '__scalar'):
            self.__scalar = Scalar()

        opts = dict(title=win)
        self.__scalar.update(self.vis, win, name, x, y,
                             opts=opts, remove=remove)

    def bar(self, win, x, rownames=None):
        if rownames is None:
            rownames = ['{}'.format(i) for i in range(x.size(0))]
        opts = dict(title=win, rownames=rownames)
        self.vis.bar(X=x, win=win, opts=opts)

    def image2d(self, win, name, img, caption=None, nrow=3):
        if not hasattr(self, '__image2d'):
            self.__image2d = Image2D()

        self.__image2d.update(self.vis, win, name, img, caption, nrow)

    def image3d(self, win, name, img):
        raise NotImplementedError

    def text(self, win, text):
        self.vis.text(text, win=win)

    def close(self, win=None):
        self.vis.close(win=win)

    def save(self):
        self.vis.save([self.vis.env])
