(define (problem learned_action_schema_fragment_online_problem)
  (:domain learned_action_schema_fragment)
  (:objects
    blue_block red_block - block
    green_cup pink_cup - cup
    green_drawer yellow_drawer - drawer
  )
  (:init
    (gripper_empty)
    (in_front_of_left green_cup yellow_drawer)
    (in_front_of_left pink_cup yellow_drawer)
    (on_left yellow_drawer)
    (on_right green_drawer)
  )
  (:init-belief
    (prob (gripper_empty) 1.0)
    (prob (in_front_of_right pink_cup green_drawer) 1.0)
    (prob (in_front_of_left green_cup yellow_drawer) 1.0)
    (prob (on_left yellow_drawer) 1.0)
    (prob (on_right green_drawer) 1.0)
    (prob (gripper_holding_block blue_block) 0.0)
    (prob (gripper_holding_block red_block) 0.0)
    (prob (gripper_holding_cup green_cup) 0.0)
    (prob (gripper_holding_cup pink_cup) 0.0)
    (prob (in_front_of_left pink_cup green_drawer) 0.0)
    (prob (in_front_of_left green_cup green_drawer) 0.0)
    (prob (in_front_of_left pink_cup yellow_drawer) 0.0)
    (prob (in_front_of_right green_cup yellow_drawer) 0.0)
    (prob (in_front_of_right pink_cup yellow_drawer) 0.0)
    (prob (in_front_of_right green_cup green_drawer) 0.0)
    (prob (on_left green_drawer) 0.0)
    (prob (on_right yellow_drawer) 0.0)
    (prob (on_top_of_left_block_drawer blue_block green_drawer) 0.0)
    (prob (on_top_of_left_block_drawer blue_block yellow_drawer) 0.0)
    (prob (on_top_of_left_block_drawer red_block green_drawer) 0.0)
    (prob (on_top_of_left_block_drawer red_block yellow_drawer) 0.0)
    (prob (on_top_of_right_cup_drawer green_cup green_drawer) 0.0)
    (prob (on_top_of_right_cup_drawer green_cup yellow_drawer) 0.0)
    (prob (on_top_of_right_cup_drawer pink_cup green_drawer) 0.0)
    (prob (on_top_of_right_cup_drawer pink_cup yellow_drawer) 0.0)
    (prob (open green_drawer) 0.0)
    (prob (open yellow_drawer) 0.0)
    (prob (contains_dark_liquid green_cup) 0.25)
    (prob (contains_dark_liquid pink_cup) 0.181818)
    (joint 0.5 (and (not (in green_drawer blue_block)) (in yellow_drawer blue_block)))
    (joint 0.5 (and (in green_drawer blue_block) (not (in yellow_drawer blue_block))))
    (joint 0.5 (and (not (in green_drawer red_block)) (in yellow_drawer red_block)))
    (joint 0.5 (and (in green_drawer red_block) (not (in yellow_drawer red_block))))
  )
  (:goal
    (or
      (and
        (not
          (in green_drawer blue_block)
        )
        (not
          (in green_drawer red_block)
        )
        (in yellow_drawer blue_block)
        (in yellow_drawer red_block)
        (not
          (open green_drawer)
        )
        (not
          (open yellow_drawer)
        )
      )
      (and
        (in green_drawer blue_block)
        (in green_drawer red_block)
        (not
          (in yellow_drawer blue_block)
        )
        (not
          (in yellow_drawer red_block)
        )
        (not
          (open green_drawer)
        )
        (not
          (open yellow_drawer)
        )
      )
    )
  )
  (:metric maximize (total-reward))
)
