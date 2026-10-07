(define (problem episode28_inferred_problem)
  (:domain learned_action_schema_fragment)
  (:objects
    yellow_drawer - drawer
  )
  (:init
    (gripper_empty)
    (on_left yellow_drawer)
    (open yellow_drawer)
  )
  (:goal
    (and (not (open yellow_drawer)))
  )
)
