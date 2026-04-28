"""300-query bank covering all 30 ESC-50 categories in level_3 of the
AVLMaps sound config.

Each query is paired with the ESC-50 category name (underscore form,
matching how it's written in `range_and_audio_meta_level_*.txt`). The
bank spans three difficulty bands per category:
  * Easy    — query mentions the source object directly.
  * Medium  — query uses domain-overlapping language but not the class word.
  * Hard    — query expresses intent only.

Distribution: ~10 queries per category. Total = 300.
"""
from __future__ import annotations

from typing import List, Tuple


# (query, esc50_category) — category in underscore form.
QUERIES_30CAT: List[Tuple[str, str]] = [
    # ---------- Interior/domestic (10 cats) ----------
    # door_wood_knock
    ("who is at the door",                          "door_wood_knock"),
    ("who is at the front entrance",                "door_wood_knock"),
    ("someone is knocking",                         "door_wood_knock"),
    ("find the visitor at the door",                "door_wood_knock"),
    ("I hear knocking",                             "door_wood_knock"),
    ("is someone trying to enter",                  "door_wood_knock"),
    ("who wants to come in",                        "door_wood_knock"),
    ("find the source of the rapping",              "door_wood_knock"),
    ("answer the door",                             "door_wood_knock"),
    ("who is tapping on the wood",                  "door_wood_knock"),

    # mouse_click
    ("someone is clicking the mouse",               "mouse_click"),
    ("I hear a mouse click",                        "mouse_click"),
    ("find the computer mouse",                     "mouse_click"),
    ("who is using the mouse",                      "mouse_click"),
    ("find the source of the clicking",             "mouse_click"),
    ("where is the cursor activity",                "mouse_click"),
    ("someone is browsing the computer",            "mouse_click"),
    ("find the desk where someone is clicking",     "mouse_click"),
    ("find the active mouse",                       "mouse_click"),
    ("locate the click sound",                      "mouse_click"),

    # keyboard_typing
    ("who is typing",                               "keyboard_typing"),
    ("where is the keyboard",                       "keyboard_typing"),
    ("where can I find the active home office",     "keyboard_typing"),
    ("find someone using a computer",               "keyboard_typing"),
    ("who is at the desk",                          "keyboard_typing"),
    ("the home office is occupied",                 "keyboard_typing"),
    ("find the workspace in use",                   "keyboard_typing"),
    ("who is doing computer work",                  "keyboard_typing"),
    ("where is the click-clack of typing",          "keyboard_typing"),
    ("find the active workstation",                 "keyboard_typing"),

    # door_wood_creaks
    ("the door is creaking",                        "door_wood_creaks"),
    ("find the creaking door",                      "door_wood_creaks"),
    ("I hear an old wooden door",                   "door_wood_creaks"),
    ("the house is settling",                       "door_wood_creaks"),
    ("find the source of the creak",                "door_wood_creaks"),
    ("something wooden is creaking",                "door_wood_creaks"),
    ("the hinges are squeaking",                    "door_wood_creaks"),
    ("find the squeaky door",                       "door_wood_creaks"),
    ("I hear creaking floorboards or door",         "door_wood_creaks"),
    ("the old door needs oil",                      "door_wood_creaks"),

    # can_opening
    ("someone is opening a can",                    "can_opening"),
    ("find the can opener",                         "can_opening"),
    ("someone popped a soda",                       "can_opening"),
    ("I hear a can being opened",                   "can_opening"),
    ("where is the snack happening",                "can_opening"),
    ("find the source of the pop sound",            "can_opening"),
    ("metallic opening sound",                      "can_opening"),
    ("someone is having a beverage",                "can_opening"),
    ("find the open beverage",                      "can_opening"),
    ("locate the can-opening sound",                "can_opening"),

    # washing_machine
    ("find the washing machine",                    "washing_machine"),
    ("which appliance might be malfunctioning",     "washing_machine"),
    ("where is the laundry running",                "washing_machine"),
    ("find the vibrating appliance",                "washing_machine"),
    ("who started the wash cycle",                  "washing_machine"),
    ("the laundry is going",                        "washing_machine"),
    ("find the source of the squeal",               "washing_machine"),
    ("which device is shaking",                     "washing_machine"),
    ("find the rumbling device",                    "washing_machine"),
    ("I hear something vibrating",                  "washing_machine"),

    # vacuum_cleaner
    ("where is the vacuum cleaner",                 "vacuum_cleaner"),
    ("find the small whining motor",                "vacuum_cleaner"),
    ("which cleaning device is running",            "vacuum_cleaner"),
    ("I hear a high-pitched motor",                 "vacuum_cleaner"),
    ("who is doing the floor cleaning",             "vacuum_cleaner"),
    ("find the appliance with a whining sound",     "vacuum_cleaner"),
    ("where is someone vacuuming",                  "vacuum_cleaner"),
    ("the floor cleaner is on",                     "vacuum_cleaner"),
    ("find the source of the buzz",                 "vacuum_cleaner"),
    ("who is vacuuming the carpet",                 "vacuum_cleaner"),

    # clock_alarm
    ("the alarm is going off",                      "clock_alarm"),
    ("find the alarm clock",                        "clock_alarm"),
    ("who set the alarm",                           "clock_alarm"),
    ("I hear an alarm",                             "clock_alarm"),
    ("turn off the alarm",                          "clock_alarm"),
    ("find the morning alarm",                      "clock_alarm"),
    ("where is the ringing",                        "clock_alarm"),
    ("find the source of the alarm beeping",        "clock_alarm"),
    ("the wake-up alarm is going",                  "clock_alarm"),
    ("locate the buzzing alarm",                    "clock_alarm"),

    # clock_tick
    ("I hear a clock ticking",                      "clock_tick"),
    ("find the ticking clock",                      "clock_tick"),
    ("where is the tick-tock",                      "clock_tick"),
    ("find the analog clock",                       "clock_tick"),
    ("locate the source of the ticking",            "clock_tick"),
    ("find the wall clock",                         "clock_tick"),
    ("where is the timepiece",                      "clock_tick"),
    ("find the slow rhythmic ticking",              "clock_tick"),
    ("an old clock is ticking",                     "clock_tick"),
    ("find the source of the tick-tick sound",      "clock_tick"),

    # glass_breaking
    ("what just shattered",                         "glass_breaking"),
    ("find the broken glass",                       "glass_breaking"),
    ("where was that loud crash",                   "glass_breaking"),
    ("something fell and broke",                    "glass_breaking"),
    ("which appliance just had a loud accident",    "glass_breaking"),
    ("find the source of the clattering",           "glass_breaking"),
    ("who broke something",                         "glass_breaking"),
    ("I heard glass shatter",                       "glass_breaking"),
    ("something just dropped and clanked",          "glass_breaking"),
    ("where is the kitchen disaster",               "glass_breaking"),

    # ---------- Human, non-speech (10 cats) ----------
    # crying_baby
    ("the baby is crying",                          "crying_baby"),
    ("what child needs comfort right now",          "crying_baby"),
    ("find the infant",                             "crying_baby"),
    ("who needs to be soothed",                     "crying_baby"),
    ("the kid is upset",                            "crying_baby"),
    ("where is the baby",                           "crying_baby"),
    ("who is wailing",                              "crying_baby"),
    ("find the source of the crying",               "crying_baby"),
    ("the toddler is fussy",                        "crying_baby"),
    ("where is the unhappy little one",             "crying_baby"),

    # sneezing
    ("someone is sneezing",                         "sneezing"),
    ("find the sneezer",                            "sneezing"),
    ("who has a cold",                              "sneezing"),
    ("I hear a sneeze",                             "sneezing"),
    ("someone said achoo",                          "sneezing"),
    ("find someone with allergies",                 "sneezing"),
    ("where is the source of the sneeze",           "sneezing"),
    ("who needs a tissue",                          "sneezing"),
    ("find the cold sufferer",                      "sneezing"),
    ("locate the sneezing sound",                   "sneezing"),

    # clapping
    ("someone is clapping",                         "clapping"),
    ("find the applause",                           "clapping"),
    ("who is celebrating",                          "clapping"),
    ("I hear clapping",                             "clapping"),
    ("find the source of the applause",             "clapping"),
    ("where are people cheering",                   "clapping"),
    ("the clapping is coming from where",           "clapping"),
    ("find the round of applause",                  "clapping"),
    ("locate the celebration",                      "clapping"),
    ("find the hands clapping together",            "clapping"),

    # breathing
    ("someone is breathing heavily",                "breathing"),
    ("find the source of the breathing",            "breathing"),
    ("I hear heavy breath",                         "breathing"),
    ("find the panting person",                     "breathing"),
    ("where is the deep breathing",                 "breathing"),
    ("someone is winded",                           "breathing"),
    ("find the audible breathing",                  "breathing"),
    ("locate the panting",                          "breathing"),
    ("who is out of breath",                        "breathing"),
    ("find the heavy breather",                     "breathing"),

    # coughing
    ("someone is coughing",                         "coughing"),
    ("find the sick person",                        "coughing"),
    ("I hear a cough",                              "coughing"),
    ("find the source of the cough",                "coughing"),
    ("someone has a sore throat",                   "coughing"),
    ("find someone hacking",                        "coughing"),
    ("who is throat-clearing",                      "coughing"),
    ("locate the coughing",                         "coughing"),
    ("find the unwell person",                      "coughing"),
    ("someone is coughing up a fit",                "coughing"),

    # footsteps
    ("someone is walking",                          "footsteps"),
    ("find the footsteps",                          "footsteps"),
    ("I hear walking",                              "footsteps"),
    ("someone is approaching",                      "footsteps"),
    ("find the source of the footsteps",            "footsteps"),
    ("who is walking around",                       "footsteps"),
    ("find the moving person",                      "footsteps"),
    ("where are those steps",                       "footsteps"),
    ("locate the walker",                           "footsteps"),
    ("someone is pacing",                           "footsteps"),

    # laughing
    ("who is laughing",                             "laughing"),
    ("find the laughter",                           "laughing"),
    ("where is the joy",                            "laughing"),
    ("someone is having fun",                       "laughing"),
    ("find the giggling",                           "laughing"),
    ("I hear chuckling",                            "laughing"),
    ("find the source of the laugh",                "laughing"),
    ("someone is amused",                           "laughing"),
    ("find the happy person",                       "laughing"),
    ("locate the laughter",                         "laughing"),

    # brushing_teeth
    ("who is brushing their teeth",                 "brushing_teeth"),
    ("who is doing their morning hygiene",          "brushing_teeth"),
    ("find the bathroom activity",                  "brushing_teeth"),
    ("someone is in the bathroom",                  "brushing_teeth"),
    ("find the morning routine in progress",        "brushing_teeth"),
    ("who is at the sink for hygiene",              "brushing_teeth"),
    ("where is the toothbrush being used",          "brushing_teeth"),
    ("find the dental care happening",              "brushing_teeth"),
    ("someone is doing oral hygiene",               "brushing_teeth"),
    ("find where teeth are being cleaned",          "brushing_teeth"),

    # snoring
    ("someone is snoring",                          "snoring"),
    ("find the sleeper",                            "snoring"),
    ("who is asleep",                               "snoring"),
    ("I hear snoring",                              "snoring"),
    ("find the source of the snore",                "snoring"),
    ("find the loud sleeper",                       "snoring"),
    ("where is the napping person",                 "snoring"),
    ("locate the heavy snorer",                     "snoring"),
    ("find someone in deep sleep",                  "snoring"),
    ("who needs a sleep apnea check",               "snoring"),

    # drinking_sipping
    ("someone is drinking",                         "drinking_sipping"),
    ("find the sipping sound",                      "drinking_sipping"),
    ("who is having a drink",                       "drinking_sipping"),
    ("I hear sipping",                              "drinking_sipping"),
    ("find the source of the gulping",              "drinking_sipping"),
    ("someone is having tea",                       "drinking_sipping"),
    ("locate the slurping",                         "drinking_sipping"),
    ("who is taking a drink",                       "drinking_sipping"),
    ("find the person enjoying a beverage",         "drinking_sipping"),
    ("where is the glass-to-mouth activity",        "drinking_sipping"),

    # ---------- Animals (10 cats) ----------
    # dog
    ("where is the dog",                            "dog"),
    ("where is the pet making noise",               "dog"),
    ("find the barking animal",                     "dog"),
    ("what creature is being loud",                 "dog"),
    ("I hear an animal in the house",               "dog"),
    ("the pet wants attention",                     "dog"),
    ("who is making barking sounds",                "dog"),
    ("locate the source of the bark",               "dog"),
    ("the pup is loud right now",                   "dog"),
    ("find where the woof is coming from",          "dog"),

    # rooster
    ("where is the rooster",                        "rooster"),
    ("find the crowing",                            "rooster"),
    ("I hear a rooster",                            "rooster"),
    ("morning rooster sound",                       "rooster"),
    ("find the cock-a-doodle-doo",                  "rooster"),
    ("find the source of the crow",                 "rooster"),
    ("the morning farm sound",                      "rooster"),
    ("who is crowing at dawn",                      "rooster"),
    ("find the cockerel",                           "rooster"),
    ("locate the rooster's crow",                   "rooster"),

    # pig
    ("where is the pig",                            "pig"),
    ("find the oinking",                            "pig"),
    ("I hear a pig",                                "pig"),
    ("barnyard pig",                                "pig"),
    ("find the snorting animal",                    "pig"),
    ("find the source of the oink",                 "pig"),
    ("locate the pig sound",                        "pig"),
    ("find the farm animal grunting",               "pig"),
    ("the pig is making noise",                     "pig"),
    ("find the squealing pig",                      "pig"),

    # cow
    ("where is the cow",                            "cow"),
    ("find the moo",                                "cow"),
    ("I hear a cow",                                "cow"),
    ("find the source of the moo",                  "cow"),
    ("find the bovine",                             "cow"),
    ("locate the cattle",                           "cow"),
    ("the cow is calling",                          "cow"),
    ("find the lowing animal",                      "cow"),
    ("where is the dairy animal",                   "cow"),
    ("find the cow's call",                         "cow"),

    # frog
    ("where is the frog",                           "frog"),
    ("find the croaking",                           "frog"),
    ("I hear a frog",                               "frog"),
    ("find the source of the croak",                "frog"),
    ("find the ribbit",                             "frog"),
    ("locate the amphibian",                        "frog"),
    ("garden frog sound",                           "frog"),
    ("find the croaking pond animal",               "frog"),
    ("the frog is calling",                         "frog"),
    ("find the leaping animal",                     "frog"),

    # cat
    ("where is the cat",                            "cat"),
    ("find the meow",                               "cat"),
    ("I hear a cat",                                "cat"),
    ("find the source of the meow",                 "cat"),
    ("where is the kitty",                          "cat"),
    ("find the feline",                             "cat"),
    ("the cat wants attention",                     "cat"),
    ("find the meowing pet",                        "cat"),
    ("locate the kitten",                           "cat"),
    ("who is purring or meowing",                   "cat"),

    # hen
    ("where is the hen",                            "hen"),
    ("find the clucking",                           "hen"),
    ("I hear a hen",                                "hen"),
    ("find the chicken",                            "hen"),
    ("barnyard hen sound",                          "hen"),
    ("find the cluck-cluck",                        "hen"),
    ("find the source of the clucking",             "hen"),
    ("locate the laying hen",                       "hen"),
    ("the chicken is making noise",                 "hen"),
    ("find the henhouse activity",                  "hen"),

    # insects
    ("where are the bugs",                          "insects"),
    ("find the buzzing insects",                    "insects"),
    ("I hear bugs",                                 "insects"),
    ("find the source of the buzz",                 "insects"),
    ("where are the cicadas",                       "insects"),
    ("find the chirping insects",                   "insects"),
    ("where are the crickets",                      "insects"),
    ("locate the insect sound",                     "insects"),
    ("find the bug noise",                          "insects"),
    ("a summer insect chorus",                      "insects"),

    # sheep
    ("where is the sheep",                          "sheep"),
    ("find the bleating",                           "sheep"),
    ("I hear a sheep",                              "sheep"),
    ("find the source of the bleat",                "sheep"),
    ("find the baa-ing animal",                     "sheep"),
    ("locate the lamb",                             "sheep"),
    ("where is the wooly animal",                   "sheep"),
    ("find the flock",                              "sheep"),
    ("the sheep is calling",                        "sheep"),
    ("find the bleating barnyard animal",           "sheep"),

    # crow
    ("where is the crow",                           "crow"),
    ("find the cawing",                             "crow"),
    ("I hear a crow",                               "crow"),
    ("find the source of the caw",                  "crow"),
    ("find the black bird",                         "crow"),
    ("find the cawing bird",                        "crow"),
    ("locate the raven",                            "crow"),
    ("find the source of the squawk",               "crow"),
    ("the crow is calling",                         "crow"),
    ("find the noisy black bird",                   "crow"),
]


# Sanity-check there are exactly 10 queries per category.
def _audit() -> None:
    from collections import Counter
    counts = Counter(c for _, c in QUERIES_30CAT)
    print(f"total queries: {len(QUERIES_30CAT)}")
    print(f"categories:    {len(counts)}")
    for cat, n in counts.most_common():
        if n != 10:
            print(f"  {cat}: {n}")


if __name__ == "__main__":
    _audit()
