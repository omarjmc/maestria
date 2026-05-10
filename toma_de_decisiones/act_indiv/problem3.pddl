(define (problem rover-problema3)
  (:domain rover-minerales)

  (:objects
    rover1 rover2 - rover
    loc1 loc2 loc3 loc4 loc5 loc8 - localidad
    mineral1 mineral2 mineral3 mineral4 - mineral
  )

  (:init
    ; Posiciones iniciales
    (en-rover rover1 loc3)
    (manos-libres rover1)
    (en-rover rover2 loc4)
    (manos-libres rover2)

    ; Minerales
    (en-mineral mineral1 loc1)
    (en-mineral mineral2 loc2)
    (en-mineral mineral3 loc1)   ; segundo mineral en loc1
    (en-mineral mineral4 loc8)

    ; Laboratorio
    (laboratorio loc5)

    ; Red de caminos
    (camino loc3 loc1)
    (camino loc1 loc3)
    (camino loc3 loc2)
    (camino loc2 loc4)
    (camino loc3 loc4)
    (camino loc4 loc3)
    (camino loc4 loc5)
    (camino loc5 loc4)

    ; Nuevos caminos hacia loc8
    (camino loc1 loc8)     ; una sola dirección
    (camino loc8 loc4)     ; salida directa a loc4
  )

  (:goal
    (and
      (analizado mineral1)
      (analizado mineral2)
      (analizado mineral3)
      (analizado mineral4)
    )
  )
)