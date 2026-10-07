(define (problem episode21_inferred_problem)
  (:domain learned_action_schema_fragment)
  (:objects
    blue_block - block
    green_drawer yellow_drawer - drawer
  )
  (:init
    (gripper_empty)
    (in blue_block green_drawer)
    (not (gripper_holding blue_block))
    (not (on_top_of blue_block yellow_drawer))
    (not (open green_drawer))
    (on_left yellow_drawer)
  )
  (:goal
    (and (on_top_of blue_block yellow_drawer))
  )
)
