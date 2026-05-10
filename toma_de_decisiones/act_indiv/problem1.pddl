(define (problem rover-problema1)
  (:domain rover-minerales)

  (:objects
    rover1 - rover
    loc1 loc2 loc3 loc4 loc5 - localidad
    mineral1 mineral2 - mineral
  )

  (:init
    ; Posición inicial del rover
    (en-rover rover1 loc3)
    (manos-libres rover1)

    ; Minerales excavados
    (en-mineral mineral1 loc1)
    (en-mineral mineral2 loc2)

    ; Laboratorio
    (laboratorio loc5)

    ; Red de caminos
    ;   loc3 <-> loc1  (bidireccional)
    (camino loc3 loc1)
    (camino loc1 loc3)
    ;   loc3 -> loc2   (una dirección)
    (camino loc3 loc2)
    ;   loc2 -> loc4   (una dirección)
    (camino loc2 loc4)
    ;   loc3 <-> loc4  (bidireccional)
    (camino loc3 loc4)
    (camino loc4 loc3)
    ;   loc4 <-> loc5  (bidireccional)
    (camino loc4 loc5)
    (camino loc5 loc4)
  )

  (:goal
    (and
      (analizado mineral1)
      (analizado mineral2)
    )
  )
)