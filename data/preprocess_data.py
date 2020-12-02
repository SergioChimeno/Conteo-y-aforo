import cv2 as cv
import os
import matplotlib.pyplot as plt
import subprocess
import shutil

VIDEOS_PATH='stanford_campus_dataset/videos'
ANNOTATIONS_PATH='stanford_campus_dataset/annotations'
POSITIVE_PATH='positive'
NEGATIVE_PATH='negative'
SLIDE_WINDOW_HEIGHT=24
SLIDE_WINDOW_WIDTH=14

def get_paths():
    """
    Returns:
        vid_path: los paths de los videos
        ann_path: los paths de las anotaciones
    """
    def get_paths_aux(path):
        paths = []
        for dir_path, dir_names, file_names in os.walk(path):
            for file_name in file_names:
                paths.append(dir_path+'/'+file_name)

        return paths

    vid_path = get_paths_aux(VIDEOS_PATH)
    ann_path=get_paths_aux(ANNOTATIONS_PATH)

    return vid_path, ann_path[1::2]


def iterar_imagenes_positivas(funcion_sobre_imagen,cantidad_imagenes=1000,anotaciones_ignoradas=100):
    """
    Función que itera sobre las imágenes positivas de los videos del dataset de stanford y aplica una función a cada una
    de ellas

    Args:
        cantidad_imagenes: cantidad de imágenes positivas a extraer
        funcion_sobre_imagen: funcion que se aplicara a cada imágen positiva
        anotaciones_ignoradas: cantidad de anotaciones que se ignoran entre 2 anotaciones válidas
    """
    clases_peatones=['"Biker"','"Pedestrian"','"Skater"']
    imagenes_procesadas=0
    flag_break=False

    for video_path, annotation_path in zip(videos_path,annotations_path):
        video = cv.VideoCapture(video_path)

        with open(annotation_path) as f:
            contador_linea=0
            for linea in f:
                if imagenes_procesadas >= cantidad_imagenes:
                    flag_break=True
                    break

                if contador_linea>=anotaciones_ignoradas:
                    campos = linea.split()

                    label= campos[9]
                    lost= int(campos[6])
                    occluded = int(campos[7])
                    generated = int(campos[8])
                    if label not in clases_peatones or lost or occluded or generated:
                        continue
                    # print(campos)
                    xmin=int(campos[1])
                    ymin = int(campos[2])
                    xmax = int(campos[3])
                    ymax = int(campos[4])
                    frame_n = int(campos[5])

                    video.set(cv.CAP_PROP_POS_FRAMES, frame_n)
                    ret, frame = video.read()
                    roi = frame[ymin:ymax, xmin:xmax]
                    funcion_sobre_imagen(roi)
                    imagenes_procesadas+=1
                    #print(imagenes_procesadas)

                    contador_linea = 0
                else:
                    contador_linea+=1

            if flag_break:
                break

        video.release()


def visualizar_shape_imagenes_positivas(tamano_muestra,anotaciones_ignoradas):
    """Mostrar el grafico de puntos, donde cada punto es (width,height) de cada imagen positiva

        Args:
            tamano_muestra: cantidad de imagenes de la muestra
            anotaciones_ignoradas: cantidad de anoraciones que se ignoran entre 2 anotaciones que sí se utilizan
    """
    w=[]
    h=[]
    def anadir_shape(img):
        w.append(img.shape[1])
        h.append(img.shape[0])
    
    iterar_imagenes_positivas(anadir_shape,cantidad_imagenes=tamano_muestra,anotaciones_ignoradas=anotaciones_ignoradas)

    average_w=sum(w)/len(w)
    average_h = sum(h) / len(h)

    plt.scatter(w,h)
    plt.plot([average_w]*2,[min(h),max(h)])
    plt.plot( [min(w), max(w)],[average_h] * 2)
    plt.title("Grafica width, height")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.show()


def guardar_imagenes_positivas(path,cantidad_imagenes=10,anotaciones_ignoradas=10000):
    """
    Función que obtine las imágenes positivas del dataset de stanford
    y las coloca en la carpata :path

    Args:
        cantidad_imagenes: cantidad de imágenes positivas a extraer
        path: donde se guardarán las imagenes resultantes
        anotaciones_ignoradas: cantidad de anotaciones que se ignoran entre 2 anotaciones válidas
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

    contador_imagen=0
    def guardar_imagen(img):
        nonlocal contador_imagen

        img = cv.resize(img,(SLIDE_WINDOW_WIDTH,SLIDE_WINDOW_HEIGHT))
        cv.imwrite(path+f'/{contador_imagen}.jpg',img)
        contador_imagen+=1

    iterar_imagenes_positivas(guardar_imagen, cantidad_imagenes=cantidad_imagenes,anotaciones_ignoradas=anotaciones_ignoradas)


def guardar_imagenes_negativas(path,cantidad_imagenes=10):
    """
    Función que obtine las imágenes negativas del dataset de stanford
    y las coloca en la carpata :path

    Args:
        path: donde se guardaran las imagenes obtenidas
        cantidad_imagenes: cantidad de imágenes positivas a extraer
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
    
    clases_peatones=['"Biker"','"Pedestrian"','"Skater"']
    imagenes_procesadas=0
    flag_break=False

    for video_path, annotation_path in zip(videos_path,annotations_path):
        video = cv.VideoCapture(video_path)

        n_frame=video.get(cv.CAP_PROP_FRAME_COUNT)-20
        video.set(cv.CAP_PROP_POS_FRAMES, n_frame)
        ret, frame = video.read()

        with open(annotation_path) as f:
            for linea in f:
                if imagenes_procesadas >= cantidad_imagenes:
                    flag_break=True
                    break

                campos = linea.split()

                n_frame_annotation = int(campos[5])
                lost = int(campos[6])
                label = campos[9]

                if n_frame_annotation != n_frame or label not in clases_peatones:
                    continue

                xmin=int(campos[1])
                ymin = int(campos[2])
                xmax = int(campos[3])
                ymax = int(campos[4])

                roi = frame[ymin:ymax, xmin:xmax]= (0,0,0)

        frame_h, frame_w = frame.shape[0:2]
        n_rows = frame_h //SLIDE_WINDOW_HEIGHT
        n_cols = frame_w // SLIDE_WINDOW_WIDTH


        for row in range(0,n_rows):
            for col in range(0,n_cols):
                if imagenes_procesadas >= cantidad_imagenes:
                    flag_break=True
                    break
                roi = frame[row*SLIDE_WINDOW_HEIGHT:row*SLIDE_WINDOW_HEIGHT+SLIDE_WINDOW_HEIGHT
                ,col*SLIDE_WINDOW_WIDTH:col*SLIDE_WINDOW_WIDTH+SLIDE_WINDOW_WIDTH]

                cv.imwrite(path+f"/{imagenes_procesadas}.jpg",roi)
                imagenes_procesadas += 1

            if flag_break:
                break

        if flag_break:
            break

        video.release()


def create_description_file(images_path,description_file_path,positivas=False):
    """Crea la description file para las imágenes positivas o negativas, que puede ser usada posteriormente por opencv_createsamples

    Args:
        images_path: path de las imágenes a incluir en la description file
        description_file_path: path donde guardar la description file
        positivas: True en caso de que la description file se cree para las imágenes positivas
    """
    if os.path.exists(description_file_path):
        os.remove(description_file_path)
    ls=os.listdir(images_path)

    for img_path in ls:

        line= images_path+"/"+img_path
        if positivas:
            line+=f' 1 0 0 {SLIDE_WINDOW_WIDTH} {SLIDE_WINDOW_HEIGHT}'

        if img_path != ls[-1]:
            line+='\n'

        with open(description_file_path,'a') as f:
            f.write(line)


def create_vector_file(vec_path,description_file_path,cantidad_imagenes=10):
    if os.path.exists(vec_path):
        os.remove(vec_path)
    subprocess.run("opencv_createsamples -vec "+vec_path+" -info "+description_file_path+f" -w {SLIDE_WINDOW_WIDTH} -h {SLIDE_WINDOW_HEIGHT} -num {cantidad_imagenes}",shell=True)


def train_cascade(output_folder,vec_path,bg_file,num_positive,num_negative,num_stages):
    subprocess.run("opencv_traincascade -data "+output_folder+" -vec "+vec_path+" -bg "+bg_file+" -numPos "+str(num_positive)+" -numNeg "+ str(num_negative)+
                    " -numStages "+str(num_stages)+f" -w {SLIDE_WINDOW_WIDTH} -h {SLIDE_WINDOW_HEIGHT} " +
                   "-featureType LBP",shell=True)



videos_path,annotations_path=get_paths()
# visualizar_shape_imagenes_positivas(10000,100)

#Obtenemos las imágenes positivas, creamos la description file y creamos la vector file
# cantidad_imagenes=10000
# anotaciones_ignoradas=100
# guardar_imagenes_positivas(POSITIVE_PATH,cantidad_imagenes=cantidad_imagenes,anotaciones_ignoradas=anotaciones_ignoradas)
# description_file_positive="info.dat"
vec_positive_file='./positive.vec'
# create_description_file(POSITIVE_PATH,description_file_positive,positivas=True)
# create_vector_file(vec_positive_file,description_file_positive,cantidad_imagenes=cantidad_imagenes)
#
# #Obtenemos las imágenes negativas y creamos su description file
bg_file = "bg.txt"
# guardar_imagenes_negativas(NEGATIVE_PATH,cantidad_imagenes=cantidad_imagenes)
# create_description_file(NEGATIVE_PATH,bg_file)

#Entrenar la cascada
pos_images=8000
train_cascade("output_entrenamiento",vec_positive_file,bg_file,pos_images,pos_images,20)
