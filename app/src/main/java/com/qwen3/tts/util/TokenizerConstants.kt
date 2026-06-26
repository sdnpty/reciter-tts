package com.qwen3.tts.util

/**
 * Token IDs shared across the Qwen3-TTS pipeline.
 * Values match the official Qwen3 vocabulary (151_xxx range).
 */
object TokenizerConstants {
    /** Padding token (also used as blank in VITS add-blank scheme). */
    const val PAD_ID = 0

    /** Beginning-of-sequence / <|im_start|> token. */
    const val BOS_ID = 151_644

    /** End-of-sequence / <|im_end|> token. Talker stops generation here. */
    const val EOS_ID = 151_645

    /**
     * First audio codec token ID. Talker outputs token IDs >= AUDIO_TOKEN_START
     * as audio frame codes; IDs below this threshold are text/control tokens.
     */
    const val AUDIO_TOKEN_START = 151_936
}
