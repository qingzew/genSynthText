# Author: Ankush Gupta
# Date: 2015

"""
Entry-point for generating synthetic text images, as described in:

@InProceedings{Gupta16,
      author       = "Gupta, A. and Vedaldi, A. and Zisserman, A.",
      title        = "Synthetic Data for Text Localisation in Natural Images",
      booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition",
      year         = "2016",
    }
"""

import numpy as np
import h5py
import os, traceback
from synthgen import *
from common import *

## Define some configuration variables:
NUM_IMG = -1 # no. of images to use for generation (-1 to use all available):
INSTANCE_PER_IMAGE = 1 # no. of times to use the same image
SECS_PER_IMG = 10 #max time per image in seconds

# path to the data-file, containing image, depth and segmentation:
DATA_PATH = 'data'
# url of the data (google-drive public file):
DATA_URL = 'http://www.robots.ox.ac.uk/~ankush/data.tar.gz'
OUT_DIR = 'results/'

def get_data():
    dbs = []
    print colorize(Color.BLUE, 'finding dbs in dir data/h5data/ ...' , bold = True)
    for root, dirs, files in os.walk('./data/h5data/'):
        for f in files:
            if f != 'README.m':
                path = os.path.join(root, f)
                print path
                dbs.append(path)

    print colorize(Color.BLUE, '------------> done', bold = True)
    print

    return dbs

def add_res_to_db(imgname, res, db):
    """
    Add the synthetically generated text image instance
    and other metadata to the dataset.
    """
    ninstance = len(res)
    for i in xrange(ninstance):
        dname = "%s_%d"%(imgname, i)
        db['data'].create_dataset(dname,data=res[i]['img'])
        db['data'][dname].attrs['charBB'] = res[i]['charBB']
        db['data'][dname].attrs['wordBB'] = res[i]['wordBB']
        db['data'][dname].attrs['txt'] = ''.join(res[i]['txt'])


def main(viz = False):
    db_files = get_data()

    for path in db_files:
        print colorize(Color.BLUE,'getting data ' + path + ' ...' , bold = True)
        db = h5py.File(path, 'r')
        print colorize(Color.BLUE, '------------> done', bold = True)

        # open the output h5 file:
        out_db = h5py.File(OUT_DIR + '/result_' + os.path.basename(path).split('_')[-1], 'w')
        out_db.create_group('/data')
        print colorize(Color.GREEN, 'Storing the output in: '+ OUT_DIR, bold = True)

        # get the names of the image files in the dataset:
        imnames = sorted(db['image'].keys())
        N = len(imnames)
        global NUM_IMG
        if NUM_IMG < 0:
            NUM_IMG = N
        start_idx, end_idx = 0, min(NUM_IMG, N)

        RV3 = RendererV3(DATA_PATH, max_time = SECS_PER_IMG)
        for i in xrange(start_idx, end_idx):
            imname = imnames[i]

            try:
                # get the image:
                img = Image.fromarray(db['image'][imname][:])
                img = np.array(img)
                # cv2.imwrite(imname, np.array(img))

                # get the pre-computed depth:
                depth = db['depth'][imname][:].T

                # get segmentation:
                seg = db['seg'][imname][:].astype('float32')
                seg = np.array(Image.fromarray(seg)).T
                area = db['seg'][imname].attrs['area']
                label = db['seg'][imname].attrs['label']

                print colorize(Color.RED,'%d of %d'%(i, end_idx - 1), bold = True)
                res = RV3.render_text(img, depth, seg, area, label,
                                    ninstance = INSTANCE_PER_IMAGE, viz = viz)
                if len(res) > 0:
                    # non-empty : successful in placing text:
                    add_res_to_db(imname, res, out_db)
                # visualize the output:
                if viz:
                    if 'q' in raw_input(colorize(Color.RED,'continue? (enter to continue, q to exit): ',True)):
                        break
            except:
                traceback.print_exc()
                print colorize(Color.GREEN,'>>>> CONTINUING....', bold=True)
                continue

        db.close()
        out_db.close()


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Genereate Synthetic Scene-Text Images')
    parser.add_argument('--viz',action='store_true',dest='viz',default=False,help='flag for turning on visualizations')
    args = parser.parse_args()
    main(args.viz)
