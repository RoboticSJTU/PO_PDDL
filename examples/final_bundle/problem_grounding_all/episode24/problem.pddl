(define (problem episode24_inferred_problem)
  (:domain learned_action_schema_fragment)
  (:objects
    blue_block - block
    yellow_drawer - drawer
  )
  (:init
    (gripper_empty)
    (in blue_block yellow_drawer)
    (not (gripper_holding blue_block))
    (not (on_top_of blue_block yellow_drawer))
    (not (open yellow_drawer))
    (on_left yellow_drawer)
  )
  (:goal
    (and (on_top_of blue_block yellow_drawer))
  )
)
