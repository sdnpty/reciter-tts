package com.qwen3.tts.util

/**
 * Нормализация текста для русских TTS-фронтендов, у которых словарь/G2P
 * понимают только кириллицу (Vosk-TTS и т.п.):
 *
 *  - целые числа -> пропись («1941» -> «одна тысяча девятьсот сорок один»);
 *  - латинские слова -> практическая транслитерация («reciter» -> «реситер»),
 *    иначе они молча выпадают из синтеза;
 *  - частые символы (№, %, §) -> слова.
 *
 * Не претендует на полную морфологию (падежи чисел не согласуются) — цель в
 * том, чтобы текст ЧИТАЛСЯ, а не пропадал.
 */
object RuNormalizer {

    // ── Числа прописью (именительный падеж, мужской род) ─────────

    private val UNITS = arrayOf(
        "ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять",
    )
    private val UNITS_FEM = arrayOf(
        "ноль", "одна", "две", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять",
    )
    private val TEENS = arrayOf(
        "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
        "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать",
    )
    private val TENS = arrayOf(
        "", "", "двадцать", "тридцать", "сорок", "пятьдесят",
        "шестьдесят", "семьдесят", "восемьдесят", "девяносто",
    )
    private val HUNDREDS = arrayOf(
        "", "сто", "двести", "триста", "четыреста", "пятьсот",
        "шестьсот", "семьсот", "восемьсот", "девятьсот",
    )

    /** (одна, две-четыре, пять+) для каждого разряда групп. */
    private val SCALES = arrayOf(
        Triple("тысяча", "тысячи", "тысяч"),
        Triple("миллион", "миллиона", "миллионов"),
        Triple("миллиард", "миллиарда", "миллиардов"),
        Triple("триллион", "триллиона", "триллионов"),
    )

    private fun pluralForm(n: Long, forms: Triple<String, String, String>): String {
        val h = n % 100
        if (h in 11..14) return forms.third
        return when (n % 10) {
            1L -> forms.first
            2L, 3L, 4L -> forms.second
            else -> forms.third
        }
    }

    private fun tripleToWords(n: Int, feminine: Boolean): List<String> {
        val words = ArrayList<String>(3)
        if (n >= 100) words.add(HUNDREDS[n / 100])
        val rest = n % 100
        when {
            rest in 10..19 -> words.add(TEENS[rest - 10])
            else -> {
                if (rest >= 20) words.add(TENS[rest / 10])
                val u = rest % 10
                if (u > 0) words.add(if (feminine) UNITS_FEM[u] else UNITS[u])
            }
        }
        return words
    }

    /** Целое (до триллионов) прописью. Больше 15 цифр — по одной цифре. */
    fun numberToWords(digits: String): String {
        val trimmed = digits.trimStart('0').ifEmpty { "0" }
        if (trimmed.length > 15) {
            return trimmed.map { UNITS[it - '0'] }.joinToString(" ")
        }
        val n = trimmed.toLong()
        if (n == 0L) return "ноль"

        val words = ArrayList<String>()
        var rest = n
        val groups = ArrayList<Int>()   // младшие -> старшие
        while (rest > 0) { groups.add((rest % 1000).toInt()); rest /= 1000 }
        for (i in groups.indices.reversed()) {
            val g = groups[i]
            if (g == 0) continue
            val feminine = i == 1   // тысячи — женский род
            words.addAll(tripleToWords(g, feminine))
            if (i > 0) words.add(pluralForm(g.toLong(), SCALES[i - 1]))
        }
        return words.joinToString(" ")
    }

    // ── Транслитерация латиницы ──────────────────────────────────

    /** Английские названия букв в русской записи — для аббревиатур. */
    private val LETTER_NAMES = mapOf(
        'a' to "эй", 'b' to "би", 'c' to "си", 'd' to "ди", 'e' to "и",
        'f' to "эф", 'g' to "джи", 'h' to "эйч", 'i' to "ай", 'j' to "джей",
        'k' to "кей", 'l' to "эл", 'm' to "эм", 'n' to "эн", 'o' to "оу",
        'p' to "пи", 'q' to "кью", 'r' to "ар", 's' to "эс", 't' to "ти",
        'u' to "ю", 'v' to "ви", 'w' to "дабл ю", 'x' to "икс", 'y' to "вай",
        'z' to "зет",
    )

    /** Частые английские слова с устоявшимся русским произношением. */
    private val EN_WORDS = mapOf(
        "the" to "зе", "of" to "оф", "and" to "энд", "is" to "из", "are" to "ар",
        "you" to "ю", "one" to "уан", "two" to "ту", "new" to "нью", "news" to "ньюс",
        "ok" to "окей", "okay" to "окей", "hello" to "хеллоу", "hi" to "хай",
        "love" to "лав", "life" to "лайф", "time" to "тайм", "like" to "лайк",
        "game" to "гейм", "name" to "нейм", "home" to "хоум", "phone" to "фон",
        "iphone" to "айфон", "google" to "гугл", "windows" to "виндоус",
        "android" to "андроид", "internet" to "интернет", "online" to "онлайн",
        "file" to "файл", "site" to "сайт", "mail" to "мейл", "email" to "имейл",
        "make" to "мейк", "made" to "мейд", "use" to "юз", "user" to "юзер",
        "mister" to "мистер", "mr" to "мистер", "mrs" to "миссис", "ms" to "мисс",
        "smart" to "смарт", "start" to "старт", "stop" to "стоп", "world" to "уорлд",
        "white" to "уайт", "black" to "блэк", "house" to "хаус", "mouse" to "маус",
        "book" to "бук", "facebook" to "фейсбук", "youtube" to "ютуб",
        "wifi" to "вай фай", "bluetooth" to "блютус",
    )

    private val TRIGRAPHS = mapOf(
        "igh" to "ай", "eau" to "о", "sch" to "ш",
    )
    private val DIGRAPHS = mapOf(
        "sh" to "ш", "ch" to "ч", "th" to "т", "ph" to "ф", "wh" to "в",
        "oo" to "у", "ee" to "и", "ck" to "к", "qu" to "кв", "kh" to "х",
        "ay" to "ей", "ai" to "ей", "ey" to "ей", "ow" to "оу", "ou" to "ау",
        "oa" to "оу", "ew" to "ью", "aw" to "о", "au" to "о",
    )
    private val LATIN = mapOf(
        'a' to "а", 'b' to "б", 'c' to "к", 'd' to "д", 'e' to "е", 'f' to "ф",
        'g' to "г", 'h' to "х", 'i' to "и", 'j' to "дж", 'k' to "к", 'l' to "л",
        'm' to "м", 'n' to "н", 'o' to "о", 'p' to "п", 'q' to "к", 'r' to "р",
        's' to "с", 't' to "т", 'u' to "у", 'v' to "в", 'w' to "в", 'x' to "кс",
        'y' to "и", 'z' to "з",
    )
    private val SOFT_C_G = setOf('e', 'i', 'y')

    /** Аббревиатура по буквам: «TTS» -> «ти ти эс», «USB» -> «ю эс би». */
    fun spellLetters(word: String): String =
        word.lowercase().mapNotNull { LETTER_NAMES[it] }.joinToString(" ")

    private val EN_VOWELS = setOf('a', 'e', 'i', 'o', 'u', 'y')

    /** Читать ли латинское слово по буквам (аббревиатуры). */
    private fun shouldSpell(word: String): Boolean {
        if (word.length == 1) return true
        val allCaps = word.all { it.isUpperCase() }
        val noVowels = word.none { it.lowercaseChar() in EN_VOWELS }
        // TTS, USB, HDMI — по буквам; NASA, LASER (гласные, длиннее) — словом.
        return noVowels || (allCaps && word.length <= 4)
    }

    /** Латинское слово -> русское произношение (словарь, буквы или транслит). */
    fun latinWordToRussian(word: String): String {
        EN_WORDS[word.lowercase()]?.let { return it }
        if (shouldSpell(word)) return spellLetters(word)
        return transliterate(word)
    }

    /** Практическая латиница -> кириллица («reciter» -> «реситер»). */
    fun transliterate(word: String): String {
        var w = word.lowercase()
        // Немое конечное 'e': like -> лайк-стиль (согласная + e на конце).
        if (w.length >= 4 && w.last() == 'e' && w[w.length - 2] !in EN_VOWELS) {
            w = w.dropLast(1)
        }
        val sb = StringBuilder(w.length + 4)
        var i = 0
        while (i < w.length) {
            if (i + 2 < w.length) {
                val tri = TRIGRAPHS[w.substring(i, i + 3)]
                if (tri != null) { sb.append(tri); i += 3; continue }
            }
            if (i + 1 < w.length) {
                val di = DIGRAPHS[w.substring(i, i + 2)]
                if (di != null) { sb.append(di); i += 2; continue }
            }
            val c = w[i]
            val next = if (i + 1 < w.length) w[i + 1] else ' '
            when {
                c == 'c' && next in SOFT_C_G -> sb.append("с")
                c == 'g' && next in SOFT_C_G -> sb.append("дж")
                else -> sb.append(LATIN[c] ?: c)
            }
            i++
        }
        return sb.toString()
    }

    // ── Общая нормализация ───────────────────────────────────────

    private val SYMBOLS = mapOf('№' to " номер ", '%' to " процентов ", '§' to " параграф ", '&' to " и ")

    fun normalize(text: String): String {
        val sb = StringBuilder(text.length + 16)
        var i = 0
        while (i < text.length) {
            val c = text[i]
            when {
                c.isDigit() -> {
                    var j = i
                    while (j < text.length && text[j].isDigit()) j++
                    sb.append(numberToWords(text.substring(i, j)))
                    i = j
                }
                c in 'a'..'z' || c in 'A'..'Z' -> {
                    var j = i
                    while (j < text.length && (text[j] in 'a'..'z' || text[j] in 'A'..'Z')) j++
                    sb.append(latinWordToRussian(text.substring(i, j)))
                    i = j
                }
                SYMBOLS.containsKey(c) -> { sb.append(SYMBOLS[c]); i++ }
                else -> { sb.append(c); i++ }
            }
        }
        return sb.toString()
    }
}
