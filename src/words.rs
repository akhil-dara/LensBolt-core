//! Finish words: one list of 1,024 easy words (10 bits each) and a forgiving matcher for typing them.

#[derive(Debug, PartialEq, Eq)]
pub enum WordError { Unknown(String), Ambiguous(String) }

/// The word for `input`, as typed on a keyboard: trimmed, any case, matched against whole words only.
/// The candidates are the words at most one mistake away (a wrong, missing or extra letter, or two
/// neighbouring letters swapped). Exactly one candidate is the answer; none is `Unknown`; several are
/// `Ambiguous`, and the sender asks for the word again. No two words are within one mistake of each
/// other, so a correct word is always accepted, and a single mistake never names a different word
/// (the intended word is always a candidate).
pub fn word_index(input: &str) -> Result<u16, WordError> {
    let s = input.trim().to_ascii_lowercase();
    let b = s.as_bytes();
    let mut found = (0..WORDS.len()).filter(|&i| one_mistake(WORDS[i].as_bytes(), b));
    match (found.next(), found.next()) {
        (Some(i), None) => Ok(i as u16),
        (None, _) => Err(WordError::Unknown(s)),
        _ => Err(WordError::Ambiguous(s)),
    }
}

/// The one word starting with `prefix` (trimmed, any case, at least two letters), for the sender's
/// visible autocomplete; None when no word or several words start that way.
pub fn complete_word(prefix: &str) -> Option<u16> {
    let p = prefix.trim().to_ascii_lowercase();
    if p.len() < 2 { return None; }
    let mut found = (0..WORDS.len()).filter(|&i| WORDS[i].starts_with(p.as_str()));
    match (found.next(), found.next()) { (Some(i), None) => Some(i as u16), _ => None }
}

/// The autocomplete the sender may show for `typed` (the letters of the word being typed so far): the one word
/// starting with them (`complete_word`), withheld when the letters equal, or are one mistake from, the start of
/// any other word (a whole word included): then the user may have been typing that other word with a slip, and
/// a suggestion would name a word they did not mean. So a suggestion is never a word other than the one meant,
/// whatever single mistake was made, in a whole word or in the letters typed so far.
pub fn suggest(typed: &str) -> Option<u16> {
    let w = complete_word(typed)?;
    let t = typed.trim().to_ascii_lowercase();
    let t = t.as_bytes();
    let near = |v: &[u8]| (t.len().saturating_sub(1).max(1)..=(t.len() + 1).min(v.len())).any(|k| one_mistake(&v[..k], t));
    if WORDS.iter().enumerate().any(|(i, v)| i != w as usize && near(v.as_bytes())) { return None; }
    Some(w)
}

/// `a` and `b` are equal or one mistake apart: one substitution, insertion, deletion, or swap of two
/// neighbouring letters (Damerau-Levenshtein distance at most 1).
fn one_mistake(a: &[u8], b: &[u8]) -> bool {
    let (long, short) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    let p = long.iter().zip(short).take_while(|(x, y)| x == y).count();
    match long.len() - short.len() {
        0 if p == long.len() => true,
        0 => long[p + 1..] == short[p + 1..]
            || (p + 1 < long.len() && long[p] == short[p + 1] && long[p + 1] == short[p] && long[p + 2..] == short[p + 2..]),
        1 => long[p + 1..] == short[p..],
        _ => false,
    }
}

/// 3-7 lowercase letters, sorted (the index is the 10-bit value). Written for LensBolt: common concrete
/// words, no two sharing their first four letters and none the start of another, no two within one
/// mistake of each other (adjacent swaps included), no homophone pairs, nothing sad, violent, medical,
/// political, insulting or a brand.
pub const WORDS: [&str; 1024] = [
    "abacus", "acrobat", "actor", "airline", "airport", "airship", "alarm", "album", "almond", "alpaca", "anchor", "angle", "animal", "ant", "anvil", "apex",
    "apple", "apricot", "apron", "arcade", "arch", "arctic", "arena", "aroma", "artist", "artwork", "ashore", "aspen", "athlete", "atlas", "atom", "attic",
    "aurora", "author", "autumn", "avenue", "avocado", "award", "badger", "bagel", "bagpipe", "bakery", "baking", "balcony", "ball", "bamboo", "banana", "banjo",
    "bank", "banner", "banquet", "barber", "barge", "barley", "barrel", "basil", "basket", "bathtub", "baton", "battery", "bay", "beach", "beaker", "beanie",
    "bear", "bedroom", "bedside", "bee", "belt", "bento", "beret", "berry", "bicycle", "bike", "bin", "birch", "bird", "bison", "blanket", "blazer",
    "blender", "blimp", "blossom", "blouse", "blue", "board", "boat", "bobcat", "bobsled", "boiler", "bonfire", "bonnet", "bonsai", "book", "bottle", "boulder",
    "bouquet", "bowling", "bracket", "braid", "branch", "brave", "bread", "breeze", "brick", "bright", "brioche", "bristle", "bronze", "broom", "broth", "brownie",
    "brush", "bubble", "bucket", "budgie", "buffalo", "buggy", "bulb", "bulldog", "bundle", "bunny", "burrow", "bus", "butter", "buzzer", "cabbage", "cabinet",
    "cactus", "calf", "camel", "camp", "canal", "candy", "canoe", "canteen", "canvas", "canyon", "capsule", "captain", "caramel", "card", "cargo", "carpet",
    "carton", "cascade", "cashew", "castle", "cat", "cave", "cedar", "ceiling", "celery", "cellar", "cement", "cereal", "chair", "chalk", "channel", "chapter",
    "charm", "checker", "cheddar", "cheetah", "chef", "cherry", "chess", "chicken", "chili", "chimney", "chirp", "chisel", "chorus", "chunk", "churro", "cinder",
    "cinema", "circus", "citrus", "city", "clam", "clean", "cliff", "climber", "clip", "clock", "closet", "clothes", "cloud", "clover", "clown", "coaster",
    "cobalt", "cobra", "cobweb", "cocoa", "coffee", "collie", "column", "comb", "comet", "comic", "compass", "concert", "condor", "cone", "console", "cookie",
    "cooler", "copper", "coral", "corn", "cosmos", "costume", "cotton", "couch", "counter", "cowboy", "coyote", "crab", "cradle", "crane", "crayon", "cream",
    "creek", "crepe", "cricket", "crimson", "crispy", "crochet", "crumb", "crust", "crystal", "cube", "cumin", "cupcake", "cupid", "curry", "curtain", "cushion",
    "cycle", "daisy", "dance", "dawn", "daytime", "deer", "delta", "denim", "desk", "dessert", "dew", "dial", "diamond", "diary", "diner", "dingo",
    "disco", "dog", "doll", "dolphin", "domino", "donkey", "donut", "door", "dove", "dragon", "drama", "drawer", "dress", "drill", "drum", "dryer",
    "duck", "dune", "dust", "eagle", "earth", "easel", "echo", "eclair", "eclipse", "eel", "egg", "elk", "ember", "emblem", "emerald", "emoji",
    "emu", "engine", "eraser", "fabric", "factory", "fairway", "fajita", "falafel", "falcon", "fan", "farm", "feast", "fence", "fender", "fennel", "fern",
    "ferret", "fiddle", "field", "fiesta", "fig", "film", "finch", "firefly", "fish", "flannel", "flask", "fleece", "flint", "florist", "flower", "fluffy",
    "flute", "foal", "folder", "forest", "fork", "fossil", "fox", "frame", "freckle", "freezer", "fresh", "fridge", "friend", "frog", "frost", "fudge",
    "fuzzy", "gadget", "galaxy", "gallery", "game", "garage", "garden", "garlic", "gazebo", "gecko", "gelato", "gem", "gentle", "gerbil", "geyser", "gift",
    "ginger", "giraffe", "glacier", "glass", "glider", "glitter", "glove", "goblet", "goggles", "golden", "golf", "gondola", "goose", "gopher", "gown", "granola",
    "grape", "grater", "gravel", "green", "grizzly", "grocery", "guava", "guitar", "gull", "gumball", "gumdrop", "hairpin", "halibut", "hallway", "halo", "hamlet",
    "hammer", "hamster", "handbag", "hanger", "happy", "harmony", "harp", "harvest", "hawk", "hazel", "heater", "hedge", "helium", "helmet", "helper", "hen",
    "herb", "heron", "highway", "hiker", "hill", "hippo", "hive", "hockey", "holiday", "home", "honey", "hoodie", "hoop", "horizon", "hornet", "hotel",
    "house", "hula", "hut", "hydrant", "hyena", "iceberg", "icicle", "icing", "igloo", "iguana", "indigo", "inn", "insect", "iris", "iron", "island",
    "ivory", "ivy", "jacket", "jade", "jaguar", "jam", "jasmine", "jazz", "jeans", "jelly", "jersey", "jester", "jet", "jewel", "jigsaw", "jogger",
    "journal", "jug", "juice", "jukebox", "jumbo", "jumper", "jungle", "kale", "karaoke", "kayak", "kelp", "kernel", "ketchup", "kettle", "key", "kilt",
    "kimono", "kitchen", "kite", "kitten", "kiwi", "koala", "label", "ladder", "ladle", "ladybug", "lagoon", "lake", "lamb", "lantern", "laptop", "lasagna",
    "laser", "laundry", "leaf", "leash", "leather", "leek", "lemon", "lemur", "lens", "lentil", "leopard", "lettuce", "lever", "library", "lid", "lighter",
    "lilac", "lily", "linen", "lion", "lizard", "llama", "lobby", "lobster", "locker", "lodge", "loft", "lookout", "lotus", "lounge", "lucky", "luggage",
    "lullaby", "lumber", "lyric", "macaw", "machine", "magenta", "magic", "magnet", "magpie", "mailbox", "mammoth", "manatee", "mango", "mansion", "mantis", "map",
    "marble", "marina", "marker", "marlin", "marmot", "maroon", "marsh", "mascot", "maze", "meadow", "medal", "meerkat", "melon", "memo", "menu", "mermaid",
    "mesh", "message", "meteor", "mighty", "milk", "minivan", "minnow", "mint", "mirror", "misty", "mixer", "mocha", "mole", "monarch", "moon", "mosaic",
    "moss", "moth", "motor", "movie", "mural", "museum", "music", "mustard", "nacho", "napkin", "narwhal", "navy", "nebula", "necktie", "nectar", "needle",
    "nephew", "nest", "nickel", "nomad", "note", "novel", "nozzle", "nugget", "nutmeg", "nylon", "oasis", "oatmeal", "oboe", "ocean", "octagon", "octopus",
    "office", "olive", "onion", "opal", "opera", "orange", "orbit", "orca", "orchid", "oregano", "organ", "origami", "ostrich", "otter", "outback", "outfit",
    "oven", "owl", "oxygen", "oyster", "ozone", "package", "padlock", "pagoda", "paint", "pajamas", "palette", "palm", "pancake", "panda", "panel", "panther",
    "papaya", "paper", "paprika", "parade", "parcel", "park", "parrot", "parsley", "party", "pasta", "path", "patio", "peacock", "peak", "peanut", "pearl",
    "pebble", "pecan", "pelican", "pencil", "pendant", "penguin", "pepper", "perch", "perfume", "pesto", "petal", "phone", "photo", "piano", "piccolo", "pickle",
    "picnic", "picture", "pier", "pigeon", "piglet", "pigtail", "pillow", "pilot", "pinata", "pinball", "pink", "pipe", "pita", "pitcher", "pixel", "pixie",
    "pizza", "plane", "plastic", "plateau", "player", "plaza", "pliers", "plum", "plywood", "podium", "poem", "polka", "pollen", "poncho", "pony", "poodle",
    "pool", "popcorn", "port", "possum", "poster", "potato", "pottery", "powder", "prairie", "prawn", "pretzel", "printer", "prism", "prune", "pudding", "puffin",
    "pulley", "pumpkin", "puppy", "purple", "purse", "puzzle", "pyramid", "quail", "quartz", "quick", "quilt", "quiz", "quokka", "rabbit", "raccoon", "radar",
    "radish", "raft", "rain", "raisin", "rapids", "rattle", "raven", "ravioli", "recipe", "record", "red", "reef", "relay", "remote", "reptile", "rhino",
    "ribbon", "rice", "ring", "ripple", "risotto", "river", "road", "robin", "robot", "rocket", "rodeo", "roof", "rooster", "rose", "rowboat", "rowing",
    "rubber", "ruby", "ruler", "runway", "saddle", "safari", "saffron", "sage", "sail", "salad", "salmon", "salsa", "salt", "samba", "sandal", "sapling",
    "sardine", "satin", "sauce", "savanna", "saw", "scale", "scarf", "scenery", "school", "scooter", "scout", "screen", "scroll", "scuba", "seabird", "seagull",
    "seal", "seaside", "seaweed", "sedan", "seesaw", "sensor", "sesame", "shadow", "shampoo", "shark", "sheep", "shelf", "shiny", "ship", "shirt", "shoe",
    "shorts", "shovel", "shrimp", "shrub", "shutter", "sidecar", "signal", "silly", "silo", "silver", "simple", "sitcom", "skate", "sketch", "skillet", "sky",
    "sled", "sleeve", "slide", "slipper", "slope", "sloth", "smooth", "snack", "snake", "snapper", "sneaker", "snorkel", "snow", "soapbox", "soccer", "sock",
    "sofa", "song", "sorbet", "soup", "sparrow", "spatula", "speedy", "spicy", "spider", "spinach", "spiral", "splash", "spoke", "sponge", "spoon", "spring",
    "sprout", "spruce", "squash", "squid", "stadium", "stairs", "stamp", "stapler", "star", "station", "steady", "steel", "stencil", "stew", "sticker", "stone",
    "stool", "stork", "straw", "stream", "stripe", "strong", "strudel", "studio", "subway", "sugar", "suit", "summit", "sunbeam", "sundae", "sunrise", "sunset",
    "surf", "sushi", "swallow", "swan", "sweater", "sweet", "swift", "swim", "swing", "swirl", "switch", "syrup", "tabby", "table", "taco", "tadpole",
    "tailor", "tape", "tart", "tassel", "tasty", "taxi", "tea", "teddy", "tempo", "tennis", "tent", "terrace", "thimble", "thorn", "thread", "thunder",
    "tiara", "ticket", "tie", "tiger", "timber", "tinsel", "tiny", "tiptoe", "toast", "toddler", "tofu", "toggle", "token", "tomato", "tongs", "toolbox",
    "topaz", "topping", "torch", "toucan", "towel", "toy", "track", "trail", "trapeze", "tray", "tree", "trinket", "trio", "tripod", "trolley", "trophy",
    "trout", "truffle", "trumpet", "trunk", "tub", "tugboat", "tulip", "tumbler", "tuna", "tundra", "tunnel", "turbine", "turnip", "turtle", "tutu", "tuxedo",
    "tweed", "twig", "twine", "ukulele", "umpire", "unicorn", "uniform", "vacuum", "valley", "valve", "vanilla", "vase", "velvet", "vendor", "vessel", "village",
    "vinegar", "vintage", "violet", "visor", "volcano", "voyage", "wafer", "waffle", "wagon", "walkway", "wallet", "walnut", "walrus", "wand", "wasabi", "washer",
    "watch", "wedding", "well", "wetland", "wetsuit", "whale", "wheat", "wheel", "whistle", "widget", "wildcat", "window", "winter", "wire", "wombat", "wonton",
    "wood", "worm", "wrapper", "wren", "yak", "yarn", "yeti", "yoga", "yogurt", "young", "zebra", "zesty", "zigzag", "zinc", "zipper", "zoo",
];

#[cfg(test)]
mod tests {
    use super::*;
    /// The first four letters (a 3-letter word is its own key).
    fn key(w: &str) -> &str { &w[..w.len().min(4)] }
    /// Damerau-Levenshtein distance (adjacent swaps count as one mistake), kept apart from the matcher.
    fn dist(a: &str, b: &str) -> usize {
        let (a, b) = (a.as_bytes(), b.as_bytes());
        let mut d = [[0usize; 16]; 16];
        for i in 0..=a.len() { d[i][0] = i; }
        for j in 0..=b.len() { d[0][j] = j; }
        for i in 1..=a.len() { for j in 1..=b.len() {
            d[i][j] = (d[i - 1][j] + 1).min(d[i][j - 1] + 1).min(d[i - 1][j - 1] + (a[i - 1] != b[j - 1]) as usize);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] { d[i][j] = d[i][j].min(d[i - 2][j - 2] + 1); }
        } }
        d[a.len()][b.len()]
    }
    /// Every single mistake on `w`: each substitution, insertion, deletion and adjacent swap, once each.
    fn mistakes(w: &str) -> Vec<String> {
        let b = w.as_bytes();
        let mut v = Vec::new();
        for i in 0..=b.len() {
            for c in b'a'..=b'z' {
                if i < b.len() { let mut x = b.to_vec(); x[i] = c; v.push(x); }
                let mut x = b.to_vec(); x.insert(i, c); v.push(x);
            }
            if i < b.len() { let mut x = b.to_vec(); x.remove(i); v.push(x); }
            if i + 1 < b.len() { let mut x = b.to_vec(); x.swap(i, i + 1); v.push(x); }
        }
        let mut v: Vec<String> = v.into_iter().map(|x| String::from_utf8(x).unwrap()).filter(|x| x != w).collect();
        v.sort(); v.dedup();
        v
    }
    /// One mistake on `w`, uniform over all of them as `mistakes` lists them (26 substitutions per letter,
    /// 26 insertions per gap, one deletion per letter, one swap per neighbouring pair), never `w` itself.
    fn random_mistake(w: &str, next: &mut impl FnMut(usize) -> usize) -> String {
        let n = w.len();
        loop {
            let mut x = w.as_bytes().to_vec();
            let r = next(26 * n + 26 * (n + 1) + n + n.saturating_sub(1));
            if r < 26 * n { x[r / 26] = b'a' + (r % 26) as u8; }
            else if r < 26 * (2 * n + 1) { let r = r - 26 * n; x.insert(r / 26, b'a' + (r % 26) as u8); }
            else if r < 26 * (2 * n + 1) + n { x.remove(r - 26 * (2 * n + 1)); }
            else { let i = r - 26 * (2 * n + 1) - n; x.swap(i, i + 1); }
            let x = String::from_utf8(x).unwrap();
            if x != w { return x; }
        }
    }
    #[test]
    fn list_is_well_formed() {
        let mut seen = std::collections::HashSet::new();
        let mut keys = std::collections::HashSet::new();
        for w in WORDS {
            assert!((3..=7).contains(&w.len()) && w.bytes().all(|c| c.is_ascii_lowercase()), "{w}");
            assert!(seen.insert(w), "duplicate {w}");
            assert!(keys.insert(key(w)), "first four letters repeat: {w}");
        }
        // no word's key starts another word, so a whole word is never the start of another
        for a in WORDS { for b in WORDS { assert!(a == b || !b.starts_with(key(a)), "{a} / {b}"); } }
        assert!(WORDS.windows(2).all(|p| p[0] < p[1]), "list is sorted");
    }
    #[test]
    fn words_are_two_mistakes_apart() {
        let mut pairs = 0usize;
        for (i, a) in WORDS.iter().enumerate() { for b in &WORDS[i + 1..] {
            if a.len().abs_diff(b.len()) <= 1 { assert!(dist(a, b) >= 2, "{a} / {b}"); }
            pairs += 1;
        } }
        eprintln!("{pairs} word pairs, all two or more mistakes apart");
    }
    #[test]
    fn correct_typing_is_never_refused() {
        for (i, w) in WORDS.iter().enumerate() {
            assert_eq!(word_index(w), Ok(i as u16), "{w}");
            assert_eq!(word_index(&format!("  {}\n", w.to_uppercase())), Ok(i as u16), "{w} in capitals");
            assert_eq!(word_index(&format!("\t{}{} ", w[..1].to_uppercase(), &w[1..])), Ok(i as u16), "{w} capitalised");
        }
        eprintln!("{} words, each typed three ways, all accepted", WORDS.len());
        assert_eq!(word_index("qqqqqq"), Err(WordError::Unknown("qqqqqq".into())));
        assert_eq!(word_index(""), Err(WordError::Unknown(String::new())));
        assert_eq!(word_index(" Q-Z "), Err(WordError::Unknown("q-z".into())));
    }
    #[test]
    fn one_mistake_never_names_another_word() {
        let (mut total, mut refused) = (0usize, 0usize);
        for (i, w) in WORDS.iter().enumerate() { for x in mistakes(w) {
            total += 1;
            match word_index(&x) {
                Ok(j) if j as usize == i => {}
                Err(WordError::Ambiguous(_)) => refused += 1,
                r => panic!("{x} (one mistake on {w}) gave {r:?}"),
            }
        } }
        let pct = 100.0 * refused as f64 / total as f64;
        eprintln!("{total} single mistakes: 0 wrong, {refused} refused ({pct:.2}%)");
        assert!(pct <= 5.0, "{pct:.2}% refused");
    }
    #[test]
    fn two_mistakes_rate_is_measured() {
        // fixed-seed sample; a second mistake can name another word, which the code's check then catches
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = |n: usize| { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; (seed % n as u64) as usize };
        let (n, mut wrong, mut refused) = (200_000usize, 0usize, 0usize);
        for _ in 0..n {
            let i = next(WORDS.len());
            let once = random_mistake(WORDS[i], &mut next);
            let x = random_mistake(&once, &mut next);
            match word_index(&x) {
                Ok(j) if j as usize != i => wrong += 1,
                Err(WordError::Ambiguous(_)) => refused += 1,
                _ => {}
            }
        }
        let pct = |k: usize| 100.0 * k as f64 / n as f64;
        eprintln!("{n} two-mistake samples: {wrong} wrong ({:.2}%), {refused} refused ({:.2}%)", pct(wrong), pct(refused));
        assert!(pct(wrong) <= 1.5, "{:.2}% wrong", pct(wrong));
    }
    #[test]
    fn complete_word_names_only_a_unique_start() {
        for (i, w) in WORDS.iter().enumerate() {
            assert_eq!(complete_word(w), Some(i as u16), "{w}");
            assert_eq!(complete_word(&format!(" {} ", w.to_uppercase())), Some(i as u16), "{w} in capitals");
        }
        // a start shared by two words, or by none, completes to nothing; one letter is never enough
        let (a, b) = WORDS.windows(2).map(|p| (p[0], p[1])).find(|(a, b)| a[..2] == b[..2]).unwrap();
        let shared = a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count();
        assert_eq!(complete_word(&a[..shared]), None, "{a} / {b}");
        assert_eq!(complete_word("qq"), None);
        assert_eq!(complete_word(&WORDS[0][..1]), None);
        let unique = (0..WORDS.len()).filter(|&i| (2..WORDS[i].len()).any(|n| complete_word(&WORDS[i][..n]) == Some(i as u16))).count();
        eprintln!("{unique} of {} words complete from a shorter start", WORDS.len());
    }
    #[test]
    fn suggestion_never_names_another_word() {
        // every word and every single mistake of it, typed whole: the suggestion is nothing or the word meant
        let (mut total, mut right) = (0usize, 0usize);
        for (i, w) in WORDS.iter().enumerate() {
            // typed in full: nothing left to complete; it may be withheld (actor is one slip from the start of factory)
            assert!(matches!(suggest(w), None) || suggest(w) == Some(i as u16), "{w} typed in full suggests another word");
            for x in mistakes(w) {
                total += 1;
                match suggest(&x) {
                    None => {}
                    Some(j) if j as usize == i => right += 1,
                    Some(j) => panic!("{x} (one mistake on {w}) suggests {}", WORDS[j as usize]),
                }
            }
        }
        eprintln!("{total} single mistakes: 0 suggest another word, {right} ({:.2}%) still suggest the word meant", 100.0 * right as f64 / total as f64);
        // every start of every word, typed with one mistake (the letters so far, while typing): the suggestion is
        // nothing or the word meant
        let (mut starts, mut wrong, mut example) = (0usize, 0usize, None);
        for (i, w) in WORDS.iter().enumerate() {
            for n in 1..=w.len() {
                for x in mistakes(&w[..n]) {
                    starts += 1;
                    if let Some(j) = suggest(&x) { if j as usize != i { wrong += 1; example.get_or_insert(format!("{x} (one mistake on {}, the start of {w}) suggests {}", &w[..n], WORDS[j as usize])); } }
                }
            }
        }
        eprintln!("{starts} mistyped starts: {wrong} suggest another word");
        assert_eq!(wrong, 0, "{wrong} of {starts} mistyped starts suggest another word, e.g. {}", example.unwrap_or_default());
        // a clean start of a word is suggested unless it is, or is one mistake from, the start of another word
        // ("att" is one mistake from "ant")
        let shorter = (0..WORDS.len()).filter(|&i| (2..WORDS[i].len()).any(|n| suggest(&WORDS[i][..n]) == Some(i as u16))).count();
        eprintln!("{shorter} of {} words are suggested from a shorter start", WORDS.len());
        assert!(shorter >= 445, "{shorter} words suggested early (measured 445)");
        let attic = WORDS.iter().position(|&w| w == "attic").unwrap() as u16;
        assert_eq!(complete_word("att"), Some(attic));
        assert_eq!(suggest("att"), None, "att is one mistake from ant: no suggestion");
        assert_eq!(suggest("a"), None);
        assert_eq!(suggest(""), None);
    }
}
