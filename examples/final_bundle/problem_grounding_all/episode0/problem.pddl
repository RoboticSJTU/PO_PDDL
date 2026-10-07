(define (problem episode0_inferred_problem)
  (:domain learned_action_schema_fragment)
  (:objects
    black_box - box
    pink_cup - cup
    green_drawer yellow_drawer - drawer
  )
  (:init
    (gripper_empty)
    (in_front_of pink_cup yellow_drawer)
    (not (gripper_holding pink_cup))
    (not (on_top_of black_box yellow_drawer))
    (not (on_top_of pink_cup green_drawer))
    (on_left yellow_drawer)
    (on_top_of black_box green_drawer)
  )
  (:goal
    (and (on_top_of pink_cup green_drawer) (on_top_of black_box yellow_drawer))
  )
)
