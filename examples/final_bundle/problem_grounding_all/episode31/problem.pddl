(define (problem episode31_inferred_problem)
  (:domain learned_action_schema_fragment)
  (:objects
    blue_block - block
    green_drawer yellow_drawer - drawer
  )
  (:init
    (gripper_empty)
    (in blue_block yellow_drawer)
    (not (gripper_holding blue_block))
    (not (in blue_block green_drawer))
    (not (open green_drawer))
    (not (open yellow_drawer))
    (on_left yellow_drawer)
  )
  (:goal
    (and (in blue_block green_drawer))
  )
)
