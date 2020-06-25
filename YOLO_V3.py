import cv2
import numpy as np
import time
import sqlite3
from datetime import datetime

#Cargamos la red YOLO.
#net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg") # Original yolov3

net = cv2.dnn.readNet("yolov3-tiny.weights","yolov3-tiny.cfg") #Tiny Yolo

# Solucion al problema de la deteccion individualizada.
class Tracker: # Clase contiene la informacion de los objetos
    def __init__(self):
        self.trackedObjects = {}
    def push(self, instantTracking):
        objects = instantTracking.objects
        for obj in objects:
            distances_to = {}
            for t_idx in self.trackedObjects:
                tracked = self.trackedObjects[t_idx]
                distances_to[t_idx] = np.linalg.norm(np.array((tracked['pos'][-1][0], tracked['pos'][-1][1]))-np.array((obj['pos'][0], obj['pos'][1])))

            if len(distances_to) != 0:
                minimum_at = min(distances_to, key=distances_to.get)   # Key
                minimum_value = distances_to[minimum_at]

                if minimum_value <40:
                    # Si existe por cercania, lo añadimos al existente
                    print('Vehículo en seguimiento',)
                    self.trackedObjects[minimum_at]['pos'].append(obj['pos'])
                    self.trackedObjects[minimum_at]['ts_list'].append(obj['ts'])
                else:
                    # Si no existe por cercania, creamos uno nuevo
                    print("**Nuevo Vehículo**")
                    total_objects = len(self.trackedObjects)
                    self.trackedObjects[total_objects] = {}
                    self.trackedObjects[total_objects] = {
                                        'class_id': obj['class_id'],
                                        'pos': [obj['pos']],
                                        'ts_list': [obj['ts']],
                                        'last_ts': obj['ts']
                                        }

            else:  # Si no hay nada con lo que comparar, Tambien se crea.
                print('ADVERTENCIA: Esto solo debería verse una vez: [INICIANDO TRACKER...]')
                total_objects = len(self.trackedObjects)
                self.trackedObjects[total_objects] = {}
                self.trackedObjects[total_objects] = {
                    'class_id': obj['class_id'],
                    'pos': [obj['pos']],
                    'ts_list': [obj['ts']],
                    'last_ts': obj['ts']
                }

class InstantTracking: # Clase de seguimiento objetos
    def __init__(self):
        self.objects = []

    def push(self, class_id, pos):
        self.objects.append({'class_id': class_id, 'pos': pos, 'ts': datetime.now()})



def main():
# Abrimos el archivo como names, donde estan almacenadas las clases usadas.
    with open("coco.names","r") as f:
        classes = [line.strip() for line in f.readlines()]

        layer_names = net.getLayerNames()
        outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        colors = np.random.uniform(0, 255, size=(len(classes), 3))

        # cargar el video
        source = "Autovia.mp4"
        cap = cv2.VideoCapture(source)  # 0 for 1st webcam
        # Diferenciamos la camara que graba.
        if source == "PelayoLunes.mp4":
            video_source_id = 1
        elif source == "Autovia.mp4":
            video_source_id = 2

        # formato de guardar la salida:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        salida = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        font = cv2.FONT_HERSHEY_PLAIN
        starting_time = time.time()
        frame_id = 0
        tracker = Tracker()

        while (cap.isOpened()):
            it = InstantTracking()
            try:
                ret, frame = cap.read()  #Vamos leyendo el video frame a frame
                frame_id += 1
                height, width, channels = frame.shape


                # detecting objects
                blob = cv2.dnn.blobFromImage(frame, 0.00392, (600, 320), (0, 0, 0), True, crop=False)  #(nombre,factor escala,tamaño,true=invertir RGB )
                net.setInput(blob)
                outs = net.forward(outputlayers)

                # Mostrar informacion en pantalla y nivel de confianza.
                class_ids = []
                confidences = []
                boxes = []
                list_classID = (0, 1, 2, 3, 5, 7) #coco name: person,bicycle,car,motorbike,bus,truck
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]

                        if confidence > 0.5:
                            # object detected
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)

                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)

                            it.push(class_id, [x, y])

                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.5)

                for i in range(len(boxes)):
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        label = str(classes[class_ids[i]])
                        confidence = confidences[i]
                        color = colors[class_ids[i]]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 1, (255, 255, 255), 2)

                elapsed_time = time.time() - starting_time
                fps = frame_id / elapsed_time
                cv2.putText(frame, "FPS:" + str(round(fps, 2)), (1100, 50), font, 2, (0, 0, 0), 1)

                cv2.imshow("Image", frame)
                salida.write(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                tracker.push(it)

            except:
                print("Error al cargar el video o video finalizado")

                # cerrar la ventana / cerrar webcam
                cap.release()
                salida.release()
                cv2.destroyAllWindows()
        #Guardar datos de interes en la base de datos
        z = 0
        while z < len(tracker.trackedObjects):
            veh = tracker.trackedObjects[z]
            # Calculo de la direción
            posicion = veh['pos'][0][1] - veh['pos'][-1][1]
            if posicion < 0:
                direccion = 1
            else:
                direccion = 0
            # tiempo primera deteccion vehiculo
            ts = veh['ts_list'][0]
            # clase del vehiculo detectado
            moto = int(veh['class_id'])
            if moto == 0:
                vehicle_class_id = 3
            else:
                vehicle_class_id = int(veh['class_id'])

            z = z + 1


            # Base de Datos.

            # Conectar a la base de datos
            miConexion = sqlite3.connect("Traking.db")

            # Selecionar el cursor para realizar la consulta
            miCursor = miConexion.cursor()

            # valor de los argumentos
            argumentos = (vehicle_class_id, video_source_id, direccion, ts)

            Sql = """
            INSERT INTO Deteccion(id,vehicle_class_id, video_source_id, direccion, ts)
            VALUES (NULL,?,?,?,?)
             """
            # Realizar la consulta
            if (miCursor.execute(Sql, argumentos)):
                print("Registros guardado correctamente")
            else:
                print("Ha ocurrido un error al guardar los registros")
            # Terminamos la consulta
            miCursor.close()
            # Guardamos los cambios en la base de datos
            miConexion.commit()
            # Cerramos la conexion
            miConexion.close()

main()


