import cv2
import os
import imageio
def images2video(img_dir, res_path, fps=24):
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    shape = cv2.imread(os.path.join(img_dir, os.listdir(img_dir)[0])).shape[:2]
    videoWriter = cv2.VideoWriter(res_path, fourcc, fps, shape)
    for i in range(len(os.listdir(img_dir))):
        img_path = os.path.join(img_dir, str(i) + '.jpg')
        img = cv2.imread(img_path)
        videoWriter.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    videoWriter.release()

if __name__ == '__main__':
    img_dir = 'stop_plt'
    res_path = 'stop.gif'
    images = []
    for i in range(len(os.listdir(img_dir))):
        img_path = os.path.join(img_dir, str(i) + '.jpg')
        img = cv2.imread(img_path)
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    imageio.mimsave(f"{str('./')}/{res_path}", images, duration=0.08)
    # images2video(img_dir, res_path)