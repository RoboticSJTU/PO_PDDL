(define (problem episode12_inferred_problem)
  (:domain learned_action_schema_fragment)
  (:objects
    pink_cup - cup
    green_drawer yellow_drawer - drawer
  )
  (:init
    (gripper_empty)
    (in_front_of pink_cup yellow_drawer)
    (not (gripper_holding pink_cup))
    (not (on_top_of pink_cup green_drawer))
    (not (open yellow_drawer))
    (on_left yellow_drawer)
  )
  (:goal
    (and (on_top_of pink_cup green_drawer) (open yellow_drawer))
  )
)
