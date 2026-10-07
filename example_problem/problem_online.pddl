(define (problem learned_manipulation_fragment_online_problem)
  (:domain learned_manipulation_fragment)
  (:objects
    blue_block red_block - block
    black_box - box
    green_cup pink_cup - cup
    green_drawer yellow_drawer - drawer
  )
  (:init
    (gripper_empty)
    (in_front_of green_cup green_drawer)
    (in_front_of pink_cup yellow_drawer)
    (on_left yellow_drawer)
    (on_top_of black_box green_drawer)
  )
  (:init-belief
    (prob (gripper_empty) 1.0)
    (prob (in_front_of green_cup green_drawer) 1.0)
    (prob (in_front_of pink_cup yellow_drawer) 1.0)
    (prob (on_left yellow_drawer) 1.0)
    (prob (on_top_of black_box green_drawer) 1.0)
    (prob (gripper_holding blue_block) 0.0)
    (prob (gripper_holding green_cup) 0.0)
    (prob (gripper_holding pink_cup) 0.0)
    (prob (gripper_holding red_block) 0.0)
    (prob (in_front_of green_cup yellow_drawer) 0.0)
    (prob (in_front_of pink_cup green_drawer) 0.0)
    (prob (on_left green_drawer) 0.0)
    (prob (on_top_of black_box yellow_drawer) 0.0)
    (prob (on_top_of blue_block green_drawer) 0.0)
    (prob (on_top_of blue_block yellow_drawer) 0.0)
    (prob (on_top_of green_cup green_drawer) 0.0)
    (prob (on_top_of green_cup yellow_drawer) 0.0)
    (prob (on_top_of pink_cup green_drawer) 0.0)
    (prob (on_top_of pink_cup yellow_drawer) 0.0)
    (prob (on_top_of red_block green_drawer) 0.0)
    (prob (on_top_of red_block yellow_drawer) 0.0)
    (prob (open green_drawer) 0.0)
    (prob (open yellow_drawer) 0.0)
    (joint 0.5 (and (in blue_block green_drawer) (not (in blue_block yellow_drawer))))
    (joint 0.5 (and (not (in blue_block green_drawer)) (in blue_block yellow_drawer)))
    (joint 0.5 (and (in red_block green_drawer) (not (in red_block yellow_drawer))))
    (joint 0.5 (and (not (in red_block green_drawer)) (in red_block yellow_drawer)))
    (prob (has_dark_liquid green_cup) 0.5)
    (prob (has_dark_liquid pink_cup) 0.5)
  )
  (:goal
    (or
      (and
        (not
          (in blue_block green_drawer)
        )
        (in blue_block yellow_drawer)
        (not
          (in red_block green_drawer)
        )
        (in red_block yellow_drawer)
        (not
          (open green_drawer)
        )
        (not
          (open yellow_drawer)
        )
      )
      (and
        (in blue_block green_drawer)
        (not
          (in blue_block yellow_drawer)
        )
        (in red_block green_drawer)
        (not
          (in red_block yellow_drawer)
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
