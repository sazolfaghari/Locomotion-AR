import csv
import math
from datetime import timedelta, datetime, date
from decimal import Decimal
import os
from os import path
from PIL import Image, ImageDraw
import numpy as np


def gochoo_images(TempoRisultati,row_query,results):
    i= 0
    
    while i < row_query - 1:  
        sensor=TempoRisultati[i][4].strip()
        
        if sensor== 'm051':  results[0, i] = 1.;
        elif sensor== 'd011': results[1, i] = 1.;
        elif sensor=='m018':  results[2, i] = 1.;
        elif sensor=='d015':  results[3, i] = 1.;
        elif sensor=='d014':  results[4, i] = 1.;
        elif sensor=='d007':  results[5, i] = 1.;
        elif sensor=='d016':  results[6, i] = 1.;
        elif sensor=='m017':  results[7, i] = 1.;
        elif sensor== 'm016':  results[8, i] = 1.;
        elif sensor== 'm015':  results[9, i] = 1.;
        elif sensor== 'm014':  results[10, i] = 1.;
        elif sensor== 'm013': results[11, i] = 1.;
        elif sensor== 'm012': results[12, i] = 1.;
        elif sensor== 'd002': results[13, i] = 1.;
        elif sensor== 'm011':results[14, i] = 1.;
        elif sensor== 'm010':results[15, i] = 1.;
        elif sensor== 'm009':results[16, i] = 1.;
        elif sensor== 'm008': results[17, i] = 1.;
        elif sensor== 'm005': results[18, i] = 1.;
        elif sensor== 'm004': results[19, i] = 1.;
        elif sensor== 'm006': results[20, i] = 1.;
        elif sensor== 'm003': results[21, i] = 1.;
        elif sensor== 'm002': results[22, i] = 1.;
        elif sensor== 'm007': results[23, i] = 1.;
        elif sensor== 'd013': results[24, i] = 1.;
        elif sensor== 'm001': results[25, i] = 1.;
        elif sensor== 'm023': results[26, i] = 1.;
        elif sensor== 'm022': results[27, i] = 1.;
        elif sensor== 'm021': results[28, i] = 1.;
        elif sensor== 'm020': results[29, i] = 1.;
        elif sensor== 'm019': results[30, i] = 1.;
        elif sensor== 'd010': results[31, i] = 1.;
        elif sensor== 'd009': results[32, i] = 1.;
        elif sensor== 'd008': results[33, i] = 1.;
        elif sensor== 'd012': results[34, i] = 1.;
        elif sensor== 'm024': results[35, i] = 1.;
        elif sensor== 'd001': results[36, i] = 1.;
        elif sensor== 'm025': results[37, i] = 1.;
        elif sensor== 'm026': results[38, i] = 1.;
        elif sensor== 'm027': results[39, i] = 1.;
        elif sensor== 'm028': results[40, i] = 1.;
        elif sensor== 'm029': results[41, i] = 1.;
        elif sensor== 'd003': results[42, i] = 1.;
        elif sensor== 'm030': results[43, i] = 1.;
        elif sensor== 'm036': results[44, i] = 1.;
        elif sensor== 'm035': results[45, i] = 1.;
        elif sensor== 'm034': results[46, i] = 1.;
        elif sensor== 'm033': results[47, i] = 1.;
        elif sensor== 'm032': results[48, i] = 1.;
        elif sensor== 'm031': results[49, i] = 1.;
        elif sensor== 'm037': results[50, i] = 1.;
        elif sensor== 'd005': results[51, i] = 1.;
        elif sensor== 'm038': results[52, i] = 1.;
        elif sensor== 'm039': results[53, i] = 1.;
        elif sensor== 'm040': results[54, i] = 1.;
        elif sensor== 'd006': results[55, i] = 1.;
        elif sensor== 'm041': results[56, i] = 1.;
        elif sensor== 'm042': results[57, i] = 1.;
        elif sensor== 'm043': results[58, i] = 1.;
        elif sensor== 'd004': results[59, i] = 1.;
        elif sensor== 'm044': results[60, i] = 1.;
        elif sensor== 'm050': results[61, i] = 1.;
        elif sensor== 'm049': results[62, i] = 1.;
        elif sensor== 'm048': results[63, i] = 1.;
        elif sensor== 'm047': results[64, i] = 1.;
        elif sensor== 'm046': results[65, i] = 1.;
        elif sensor== 'm045': results[66, i] = 1.;
        i +=1
    return results

# piano terra (nella mappa è quello a destra)
def createImages(cur):
    '''if path.exists('content/section_all.csv'):
        os.remove('content/section_all.csv')'''

    # Massimo valore x e y per calcolo fattore di scala
    xmax = "SELECT MAX(x) FROM sensor_locations"
    ymax = "SELECT MAX(y) FROM sensor_locations"
    cur.execute(xmax)
    x = cur.fetchone()
    cur.execute(ymax)
    y = cur.fetchone()

    # Variabili immagine e altre
    
    image_width = 67
    image_height = 80
    
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
                         WHERE floor = 1"
    cur.execute(vistaPosizioniTempo)
    print("movimento_pos_tempo creata correttamente")


    queryAttività = "SELECT * FROM activities WHERE patient in  \
            (select distinct patient_id from participants where diagnosis_id not in (1,2))"
    cur.execute(queryAttività)
    attività = cur.fetchall()
    n_attività = cur.rowcount

    start = 0
    id_img = 2321
    segment_duration = 180 #120 zero overlapping
    overlap = 36
    SENSOR_LIST =['m051','d011','m018','d015','d014','d007','d016','m017','m016','m015','m014', 'm013','m012','d002','m011','m010','m009','m008','m005','m004','m006','m003','m002','m007','d013','m001','m023','m022','m021','m020', 'm019','d010','d009','d008', 'd012','m024','d001', 'm025','m026','m027','m028','m029','d003','m030','m036','m035','m034','m033','m032','m031','m037','d005','m038','m039','m040','d006','m041','m042','m043','d004','m044','m050','m049','m048','m047', 'm046','m045']
    saveFile ='images_gochoo_new.csv'
    saveFolder = './gochoo_images/'        
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

            num_three_min_segments =abs(math.ceil(actual_duration.total_seconds()/ segment_duration))
            print(num_three_min_segments)
            
            if num_three_min_segments > 0:
                segment_start = datetime.combine(datetime.now(), posizioniTempoRisultati[0][6]) 
                segment_end = (segment_start + timedelta(seconds=segment_duration))
                breakloop = True
                scount = 0
                while scount < num_three_min_segments and breakloop:
                    id_img = id_img + 1

                    query = "SELECT * FROM movimento_pos_tempo \
                              WHERE patient = {} AND time between '{}' \
                              AND '{}' AND value in ('on','open');".format(attività[start][0], segment_start.time(),segment_end.time())
                    cur.execute(query)
                    TempoRisultati = cur.fetchall() # TempoRisultati[i][4] =sensor
                    segment_row_query = cur.rowcount
                    if segment_row_query >= 5:
                        image_array = gochoo_images(TempoRisultati,segment_row_query,np.zeros((image_width,image_height)))
          
                        
                        # SEZIONE SALVATAGGIO IMMAGINE
                        im = Image.fromarray((image_array * 255).astype('uint8'), mode='L')
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
                        if diff.total_seconds() < segment_duration:
                            breakloop=False
                        elif diff.total_seconds() >= segment_duration:
                            segment_start += timedelta(seconds=segment_duration)
                            segment_end += timedelta(seconds=segment_duration - overlap)
                        scount +=1
                    else:
                        diff = actual_end - segment_end
                        if diff.total_seconds() < segment_duration:
                            breakloop=False
                        elif diff.total_seconds() >= segment_duration:
                            segment_start += timedelta(seconds=segment_duration)
                            segment_end += timedelta(seconds=segment_duration - overlap)
                        scount +=1
                            
                    
                           
                        
        # PASSAGGIO A NUOVA ATTIVITA'
        start = start + 1
            
            