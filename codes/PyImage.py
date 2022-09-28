import csv
import math
from datetime import timedelta, datetime, date
from decimal import Decimal
import os
from os import path
from PIL import Image, ImageDraw
import numpy as np
from slope import *

def create_images(cur, event_source,segment_start,segment_end,subject,row_query,draw,scaling_factor_x,scaling_factor_y,start_activity,end_activity):
    row_counter = 0
    left_sensors = ['m004', 'm005', 'm003', 'm006', 'm002', 'm007', 'm001', 'm023', 'm022', 'm021', 'm026', 'm025','m024']
    center_sensors = ['m011', 'm010', 'm009', 'm008', 'm019', 'm020']
    right_sensors = ['m012', 'm013', 'm014', 'm015', 'm016', 'm017', 'm018', 'm051']
    color_pos_sensors = [(153, 76, 0), (153,153,255), (255, 255, 255)]  # marrone, fili, bianco
    color = (153,153,255)
    velocita = 0
    green = 39

    secondiPazienteFermo = 2  # 2 come esempio, ma può essere cambiato'''
    while row_counter < row_query - 1:
    
        if row_counter > 0:
            velocitaPrecedente = velocita
            secondiVelocita = event_source[row_counter - 1][2].total_seconds()
            if secondiVelocita == 0:
                secondiVelocita = 1
            velocita = math.sqrt(
                math.pow(event_source[row_counter][0] - event_source[row_counter - 1][0], 2) + math.pow(
                    event_source[row_counter][1] - event_source[row_counter - 1][1], 2)) / \
                       secondiVelocita
            if velocita > velocitaPrecedente:
                green += 50
            else:
                if not velocita == velocitaPrecedente:
                    green -= 50

        draw.line(
            (event_source[row_counter][0] * scaling_factor_x, event_source[row_counter][1] * scaling_factor_y,
             event_source[row_counter + 1][0] *
             scaling_factor_x, event_source[row_counter + 1][1] * scaling_factor_y), fill=(11, green, 184))
             
        # Colore diverso per posizione sensore nella casa (destra, centro, sinistra)
        if(event_source[row_counter][4] in left_sensors):
            color = color_pos_sensors[0]
        elif(event_source[row_counter][4] in center_sensors):
                color = color_pos_sensors[1]
        elif(event_source[row_counter][4] in right_sensors):
                    color = color_pos_sensors[2]

        draw.point(
            (
                event_source[row_counter][0] * scaling_factor_x,
                event_source[row_counter][1] * scaling_factor_y),
            fill=color)

        if (event_source[row_counter+1][4] in left_sensors):
            color = color_pos_sensors[0]
        elif (event_source[row_counter+1][4] in center_sensors):
                color = color_pos_sensors[1]
        elif (event_source[row_counter+1][4] in right_sensors):
                    color = color_pos_sensors[2]

        draw.point(
            (
                event_source[row_counter+1][0] * scaling_factor_x,
                event_source[row_counter+1][1] * scaling_factor_y),
            fill=color)
                      
            
        # Punti in cui stanno fermi
        if event_source[row_counter][2] > timedelta(seconds=secondiPazienteFermo):
            draw.point(
                (
                    event_source[row_counter][0] * scaling_factor_x,
                    event_source[row_counter][1] * scaling_factor_y),
                fill=(245, 66, 66)) #RGB = red
                
        # sharp angle
        if row_counter +2 < row_query:
            # Punti in cui la direzione del paziente cambia di un angolo di ampiezza > 90°
            slopeAngle = getAngle(event_source[row_counter][0], event_source[row_counter + 1][0], event_source[row_counter + 2][0],
                               event_source[row_counter][1], event_source[row_counter + 1][1],event_source[row_counter + 2][1])
            #print(slopeAngle)
            if slopeAngle >= 90:
                draw.point(
                    (
                        event_source[row_counter][0] * scaling_factor_x,
                        event_source[row_counter][1] * scaling_factor_y),
                    fill=(0, 0, 0)) #RGB = black
        
        #Door
        if 'd' in event_source[row_counter][4]:
            draw.point((event_source[row_counter][0] * scaling_factor_x, event_source[row_counter][1] * scaling_factor_y), (255, 255, 0))
        
        #objects
        
        queryOggetti = "SELECT *\
                            FROM use_objects_positions \
                            WHERE patient = {} AND time between '{}' \
                            AND '{}';".format(subject, start_activity,end_activity)
     
                        
        cur.execute(queryOggetti)
        obj_source = cur.fetchall()
        row_count= cur.rowcount
        # Ci sono solo 4 sensori oggetti che vengono usati (per vederli fare select distinct sensor from used_objects;), quindi l'array di colori sarà di dimensione 4
        arrayColori = [(36, 173, 9), (237, 123, 17), (242, 0, 255),
                       (237, 147,186)]  # 0 - i001 (verde); 1 - i002 (arancione); 2 - i006 (fucsia); 3 - i010 (rosa)
        if row_count > 0 :
            k = 0
            while k < cur.rowcount:
                obj_stime = datetime.combine(datetime.now(), obj_source[k][5])
                if obj_stime >= segment_start and  obj_stime <= segment_end:
                    obj_sensor = obj_source[k][1].lower()
                    if obj_sensor == "i001":
                        print('green')
                        draw.point(
                            ((obj_source[k][3] + Decimal(-0.2)) * scaling_factor_x,
                             (obj_source[k][4]) * scaling_factor_y),
                            arrayColori[0])
                    if obj_sensor == "i002":
                        print('arancione')
                        draw.point(
                            ((obj_source[k][3]) * scaling_factor_x,
                             (obj_source[k][4] + Decimal(+0.2)) * scaling_factor_y),
                            arrayColori[1])
                    if obj_sensor == "i006":
                        print('fucsia')
                        draw.point(
                            ((obj_source[k][3] + Decimal(+0.2)) * scaling_factor_x,
                             (obj_source[k][4]) * scaling_factor_y),
                            arrayColori[2])
                    if obj_sensor == "i010":
                        print('rosa')
                        draw.point(
                            ((obj_source[k][3]) * scaling_factor_x,
                             (obj_source[k][4] + Decimal(-0.2)) * scaling_factor_y),
                            arrayColori[3])

                k = k + 1
    
        row_counter +=1
    return draw

# piano terra (nella mappa è quello a destra)
def main(cur):
    '''if path.exists('content/section_all.csv'):
        os.remove('content/section_all.csv')'''
    # Massimo valore x e y per calcolo fattore di scala
    xmax = "SELECT MAX(x) FROM sensor_locations "
    ymax = "SELECT MAX(y) FROM sensor_locations"
    cur.execute(xmax)
    x_max = cur.fetchone()
    cur.execute(ymax)
    y_max = cur.fetchone()
    
    # Variabili immagine e altre
    image_width = 100
    image_height = 130
    scaling_factor_x = image_width / x_max[0]
    scaling_factor_y = image_height / y_max[0]

    # first_sensors
    queryVistaAppoggio = "CREATE VIEW first_sensors AS SELECT * FROM events_no_duplicates \
                         WHERE value in ('on','open','present') and patient in  \
                        (select distinct patient_id from participants where diagnosis_id in (3,4,8)) \
                         ORDER BY patient, time;"
    cur.execute(queryVistaAppoggio)
    print("first_sensors creata correttamente")
    
     # movimento_pos_tempo Creazione vista
    vistaPosizioniTempo = "CREATE VIEW movimento_pos_tempo AS SELECT x, y, (lead(time,1) OVER (order by time) - time) AS \
                         delta_time, value, sensor, patient,time FROM first_sensors JOIN sensor_locations ON sensor_id = sensor\
                         WHERE floor = 0"
    cur.execute(vistaPosizioniTempo)
    print("movimento_pos_tempo creata correttamente")
    
    queryPosizioniOggettiMovimento = "CREATE VIEW use_objects_positions AS \
                                                        SELECT patient,sensor,prev_sensor, x, y, time \
                                                        FROM (SELECT first_sensors.*, LAG(sensor) OVER (ORDER BY patient) \
                                                        AS prev_sensor FROM first_sensors) first_sensors \
                                                        JOIN sensor_locations ON prev_sensor = sensor_id \
                                                        WHERE (prev_sensor LIKE 'm%' OR prev_sensor LIKE 'd%') \
                                                        AND value = 'present' AND floor=0"

    cur.execute(queryPosizioniOggettiMovimento)

    print("queryPosizioniOggettiMovimento creata correttamente")


    queryAttività = "SELECT * FROM activities WHERE patient in  \
            (select distinct patient_id from participants where diagnosis_id not in (1,2))"
    cur.execute(queryAttività)
    attività = cur.fetchall()
    n_attività = cur.rowcount

    start = 0
    id_img = -1
    saveFolder = './Images/''
    if not os.path.exists(saveFolder_trajectory):
        # Create a new directory because it does not exist 
        os.makedirs(saveFolder)
        
    saveFile = 'section_all.csv'
    segment_TH = 180
    overlap  = 36 #0.2 overlap

    while start < n_attività :
        print('person_num: '+ str(attività[start][0]))
        # SEZIONE DISEGNO MOVIMENTI
        queryPosizioniTempo = "SELECT * FROM movimento_pos_tempo \
                              WHERE patient = {} AND time between '{}' \
                              AND '{}' AND value in ('on','open');".format(attività[start][0], attività[start][2],
                                                                           attività[start][3])
        cur.execute(queryPosizioniTempo)
        posizioniTempoRisultati = cur.fetchall()
        row_query = cur.rowcount
        
        if (row_query >= 10):
            # INIZIALIZZAZIONE IMMAGINE
            print(row_query)

            im = Image.new('RGB', (image_width, image_height), (0,0,0))
            draw = ImageDraw.Draw(im)
            actual_start= datetime.combine(datetime.now(), attività[start][2]) 
            actual_end = datetime.combine(datetime.now(), attività[start][3])
            actual_duration = actual_end - actual_start

            num_three_min_segments =abs(math.ceil(actual_duration.total_seconds()/ segment_TH))
            print(num_three_min_segments)
            
            if num_three_min_segments > 0:
                segment_start = datetime.combine(datetime.now(), posizioniTempoRisultati[0][6]) 
                segment_end = (segment_start + timedelta(seconds=segment_TH))
                breakloop = True
                scount = 0
                while scount < num_three_min_segments and breakloop:
                    

                    query = "SELECT * FROM movimento_pos_tempo \
                              WHERE patient = {} AND time between '{}' \
                              AND '{}' AND value in ('on','open');".format(attività[start][0], segment_start.time(),segment_end.time())
                    cur.execute(query)
                    TempoRisultati = cur.fetchall() # TempoRisultati[i][4] =sensor
                    segment_row_query = cur.rowcount
                    
                    if segment_row_query >= 5:
                        id_img = id_img + 1
                        im = Image.new('RGB', (image_width, image_height), (200, 200, 200))
                        draw = ImageDraw.Draw(im)

                        draw = create_images(cur, TempoRisultati,segment_start,segment_end,\
                                            attività[start][0], segment_row_query,draw, scaling_factor_x,scaling_factor_y, attività[start][2], attività[start][3])

                        # SEZIONE SALVATAGGIO IMMAGINE
                        #im = Image.fromarray((image_array * 255).astype('uint8'), mode='L')
                        im = im.transpose(Image.FLIP_TOP_BOTTOM)
                        im.save(saveFolder +  str(id_img) + "_"
                                + str(attività[start][0]) + "_"
                                + str(attività[start][1]) + "_"
                                + str(segment_start.time()).replace(":", ".") + "_"
                                + str(segment_end.time()).replace(":", ".") + "_.png",'PNG')
                                
                        print("Immagine " + str(id_img) + " creata")
                        with open(saveFile, 'a', newline="") as csvfile:
                            fieldnames = ['id_immagine', 'id_paziente', 'ora_inizio', 'ora_fine', 'id_attività']
                            writer = csv.DictWriter(csvfile, fieldnames)
                            writer.writerow(
                                {'id_immagine': str(id_img),
                                 'id_paziente': str(attività[start][0]),
                                 'ora_inizio': str(segment_start.time()),
                                 'ora_fine': str(segment_end.time()),
                                 'id_attività': str(attività[start][1])})
                        csvfile.close()
                        
                        diff = actual_end - segment_end
                        if diff.total_seconds() < segment_TH:
                            breakloop=False
                        elif diff.total_seconds() >= segment_TH:
                            segment_start += timedelta(seconds=segment_TH)
                            segment_end =segment_end + timedelta(seconds=segment_TH)  - timedelta(seconds=overlap)
                        scount +=1
                    else:
                        diff = actual_end - segment_end
                        if diff.total_seconds() < segment_TH:
                            breakloop=False
                        elif diff.total_seconds() >= segment_TH:
                            segment_start += timedelta(seconds=segment_TH)
                            segment_end = segment_end +timedelta(seconds=segment_TH)- timedelta(seconds=overlap)
                        scount +=1
                            
                    
                           
                        
        # PASSAGGIO A NUOVA ATTIVITA'
        start = start + 1
            
            