(define (problem episode26_inferred_problem)
  (:domain learned_action_schema_fragment)
  (:objects
    red_block - block
    green_drawer yellow_drawer - drawer
  )
  (:init
    (gripper_empty)
    (in red_block yellow_drawer)
    (not (gripper_holding red_block))
    (not (in red_block green_drawer))
    (not (open green_drawer))
    (not (open yellow_drawer))
    (on_left yellow_drawer)
  )
  (:goal
    (and (in red_block green_drawer))
  )
)
