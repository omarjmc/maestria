(define (domain rover-minerales)
  (:requirements :typing :negative-preconditions)

  (:types
    localidad mineral rover - object
  )

  (:predicates
    (en-rover ?r - rover ?l - localidad)        ; el rover está en localidad l
    (en-mineral ?m - mineral ?l - localidad)    ; el mineral m está en localidad l
    (transportando ?r - rover ?m - mineral)     ; el rover lleva el mineral m
    (laboratorio ?l - localidad)                ; localidad l tiene laboratorio
    (analizado ?m - mineral)                    ; el mineral m ya fue analizado
    (camino ?l1 - localidad ?l2 - localidad)    ; existe camino de l1 a l2
    (manos-libres ?r - rover)                   ; el rover no carga nada
  )

  ; ------------------------------------------------------------------
  ; ACCIÓN: Mover el rover de una localidad a otra
  ; ------------------------------------------------------------------
  (:action mover
    :parameters (?r - rover ?desde - localidad ?hacia - localidad)
    :precondition (and
      (en-rover ?r ?desde)
      (camino ?desde ?hacia)
    )
    :effect (and
      (not (en-rover ?r ?desde))
      (en-rover ?r ?hacia)
    )
  )

  ; ------------------------------------------------------------------
  ; ACCIÓN: Recoger un mineral en la localidad donde está el rover
  ; ------------------------------------------------------------------
  (:action recoger
    :parameters (?r - rover ?m - mineral ?l - localidad)
    :precondition (and
      (en-rover ?r ?l)
      (en-mineral ?m ?l)
      (manos-libres ?r)
    )
    :effect (and
      (transportando ?r ?m)
      (not (en-mineral ?m ?l))
      (not (manos-libres ?r))
    )
  )

  ; ------------------------------------------------------------------
  ; ACCIÓN: Entregar mineral en el laboratorio
  ; ------------------------------------------------------------------
  (:action entregar
    :parameters (?r - rover ?m - mineral ?l - localidad)
    :precondition (and
      (en-rover ?r ?l)
      (transportando ?r ?m)
      (laboratorio ?l)
    )
    :effect (and
      (not (transportando ?r ?m))
      (manos-libres ?r)
      (analizado ?m)
    )
  )
)