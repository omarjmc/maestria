(define (problem rover-problema2)
  (:domain rover-minerales)

  (:objects
    rover1 - rover
    loc1 loc2 loc3 loc4 loc5 loc6 loc7 - localidad
    mineral1 mineral2 mineral3 - mineral
  )

  (:init
    ; Posición inicial del rover
    (en-rover rover1 loc3)
    (manos-libres rover1)

    ; Minerales excavados
    (en-mineral mineral1 loc1)
    (en-mineral mineral2 loc2)
    (en-mineral mineral3 loc6)

    ; Laboratorios
    (laboratorio loc5)
    (laboratorio loc7)

    ; Red de caminos (original)
    (camino loc3 loc1)
    (camino loc1 loc3)
    (camino loc3 loc2)
    (camino loc2 loc4)
    (camino loc3 loc4)
    (camino loc4 loc3)
    (camino loc4 loc5)
    (camino loc5 loc4)

    ; Nuevas conexiones
    (camino loc5 loc6)    ; una sola dirección
    (camino loc6 loc7)    ; bidireccional
    (camino loc7 loc6)
  )

  (:goal
    (and
      (analizado mineral1)
      (analizado mineral2)
      (analizado mineral3)
    )
  )
)