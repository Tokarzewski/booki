package dev.booki.epub

import org.junit.Assert.assertEquals
import org.junit.Test

class NumberToWordsTest {

    // ---- cardinals ----

    @Test fun zeroAndOne() {
        assertEquals("zero", NumberToWords.cardinal(0))
        assertEquals("one", NumberToWords.cardinal(1))
    }

    @Test fun teens() {
        assertEquals("eleven", NumberToWords.cardinal(11))
        assertEquals("twelve", NumberToWords.cardinal(12))
        assertEquals("nineteen", NumberToWords.cardinal(19))
    }

    @Test fun tens() {
        assertEquals("twenty", NumberToWords.cardinal(20))
        assertEquals("forty-two", NumberToWords.cardinal(42))
        assertEquals("ninety-nine", NumberToWords.cardinal(99))
    }

    @Test fun hundreds() {
        assertEquals("one hundred", NumberToWords.cardinal(100))
        assertEquals("two hundred thirty-four", NumberToWords.cardinal(234))
        assertEquals("nine hundred", NumberToWords.cardinal(900))
    }

    @Test fun thousands() {
        assertEquals("one thousand", NumberToWords.cardinal(1000))
        assertEquals("one thousand two hundred thirty-four",
            NumberToWords.cardinal(1234))
        assertEquals("twelve thousand", NumberToWords.cardinal(12000))
        assertEquals("one hundred twenty-three thousand four hundred fifty-six",
            NumberToWords.cardinal(123456))
    }

    @Test fun millions() {
        assertEquals("one million", NumberToWords.cardinal(1_000_000))
        assertEquals("two million three hundred forty-five thousand six hundred seventy-eight",
            NumberToWords.cardinal(2_345_678))
    }

    @Test fun negatives() {
        assertEquals("minus five", NumberToWords.cardinal(-5))
        assertEquals("minus one hundred", NumberToWords.cardinal(-100))
    }

    // ---- years ----

    @Test fun yearTwoThousands() {
        assertEquals("two thousand", NumberToWords.year(2000))
        assertEquals("two thousand five", NumberToWords.year(2005))
        assertEquals("twenty ten", NumberToWords.year(2010))
        assertEquals("twenty twenty-six", NumberToWords.year(2026))
    }

    @Test fun yearNineteenHundreds() {
        assertEquals("nineteen hundred", NumberToWords.year(1900))
        assertEquals("nineteen twenty-three", NumberToWords.year(1923))
        assertEquals("nineteen oh five", NumberToWords.year(1905))
        assertEquals("nineteen ninety-nine", NumberToWords.year(1999))
    }

    @Test fun yearEdgeCases() {
        // Outside 1100–2099 range → cardinal
        assertEquals("eight hundred", NumberToWords.year(800))
        assertEquals("two thousand one hundred", NumberToWords.year(2100))
    }
}
