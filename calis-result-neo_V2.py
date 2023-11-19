#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 13:17:20 2023

@author: danielm
"""

import cv2
import numpy as np
import glob
import os
import time

def obtenerRespuestas(img):
    canny = cv2.Canny(img, 20, 150)
    
    kernel = np.ones((5,5), np.uint8)
    bordes = cv2.dilate(canny, kernel)
    
    contour,_ = cv2.findContours(bordes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    objetos = bordes.copy()
    cv2.drawContours(objetos, [max(contour, key=cv2.contourArea)], -1, 255, thickness=-1)
    
    output = cv2.connectedComponentsWithStats(objetos, 4, cv2.CV_32S)
    numObj = output[0]
    labels = output[1]
    stats = output[2]
    
    mascara = np.uint8(255*(np.argmax(stats[:,4][1:])+1==labels))
    
    contours,_ = cv2.findContours(mascara, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cnt = contours[0]
    
    hull = cv2.convexHull(cnt)
    puntosConvex = hull[:,0,:]
    m,n = mascara.shape
    ar = np.zeros((m,n))
    mascaraConvex = np.uint8(cv2.fillConvexPoly(ar, puntosConvex, 1))
    
    
    vertices = cv2.goodFeaturesToTrack(mascaraConvex, 4, 0.01, 20)
    
    x = vertices[:,0,0]
    y = vertices[:,0,1]
    
    vertices = vertices[:,0,:]
    
    x0 = np.sort(x)
    y0 = np.sort(y)
    
    xn = np.zeros((1,4))
    yn = np.zeros((1,4))
    
    xn = (x==x0[2])*n+(x==x0[3])*n
    yn = (y==y0[2])*m+(y==y0[3])*m
    
    verticesN = np.zeros((4,2))
    verticesN[:,0]=xn
    verticesN[:,1]=yn
    
    vertices = np.int64(vertices)
    verticesN = np.int64(verticesN)
    
    h,_ = cv2.findHomography(vertices, verticesN)
    
    im2 = cv2.warpPerspective(img, h, (n,m))
    roi = im2[:, np.uint64(0.25*n):np.uint64(0.86*n)]
    
    opciones = ["A", "B", "C", "D", "E", "N/A"]
    respuestas = []
    
    for i in range(0,26):
        pregunta = roi[np.uint64(i*(m/26)):np.uint64((i+1)*(m/26)),:]
        sumI = []
        for j in range(0,5):
            _,col = pregunta.shape
            sumI.append(np.sum(pregunta[:,np.uint64(j*(col/5)):np.uint64((j+1)*(col/5))]))
        vmin = np.ones((1,5))*np.min(sumI)
        
        if np.linalg.norm(sumI-vmin)>0.17*col*n:
            sumI.append(float("inf"))
        else:sumI.append(-1)
        
        respuestas.append(opciones[np.argmin(sumI)])

    return respuestas

#directorio de imagenes
directImg = "/Users/danielm/Documents/coding/curso de computer vision/calificación/v4"

#directorio de salida
directOut = f"/Users/danielm/Documents/coding/curso de computer vision/calificación/v4/exmsCalis {time.strftime('%Y%m%d%H%M%S')}"
os.makedirs(directOut, exist_ok=True)

#lista vacía para almacenar resultados para archivo de texto
resultados = []

#obtención de lista de imagenes en directorio
imagenes = glob.glob(os.path.join(directImg, "*.jpg"))

#iteración y lectura de imagenes en directorio
for imagen_path in imagenes:
    img = cv2.imread(imagen_path, 0)
    
    #obtener respuestas
    respuestasFormato = np.array(obtenerRespuestas(img))
    
    #obtener respuestas para archivo de texto
    respuestasResultado = obtenerRespuestas(img)
    
    #lista de respuestas correctas
    respuestasCorrectas = np.array(["B", "C", "D", "E", "B", "C", "C", "A", "C", "D", "E", "C", "C", "A", "E", "B", "C", "A", "D", "C", "B", "D", "A", "E", "C", "A"])
    
    #cómputo de calificación
    calis = str(int(10*sum(respuestasCorrectas==respuestasFormato)/26))
    
    #agregar calificación a imagen
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f"PUNTAJE: {calis}", (100, 200), font, 3, (255, 255, 255), 5, cv2.LINE_AA)
    
    #guardar imagen procesada en directorio de salida
    nombre_salida = os.path.join(directOut, os.path.basename(imagen_path))
    cv2.imwrite(nombre_salida, img)
    
    #almacenar nombre img y puntaje en lista de resultados
    resultados.append(f"{os.path.basename(nombre_salida)}: {calis} \n Respuestas: {respuestasResultado}")

#guardar lista de resultados en archivo de texto
archivo_resultados = os.path.join(directOut, "resultados.txt")
with open(archivo_resultados, "w") as f:
    for resultado in resultados:
        f.write(resultado + "\n \n \n")

print(f"Se han guardado los resultados del análisis en {archivo_resultados}")

