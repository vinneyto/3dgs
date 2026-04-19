"use client";

import styles from "./coatOfArms.module.css";

const legendItems = [
  {
    title: "Щит разума",
    text: "Основная форма герба остаётся классической, но собрана как чистая векторная схема: разум у тебя структурный, инженерный и точный.",
  },
  {
    title: "Орбита и звёзды",
    text: "Кольцо и небесные точки добавляют научную фантастику: не побег от реальности, а привычку думать на масштабах больше текущего дня.",
  },
  {
    title: "Фигурные скобки",
    text: "Две светлые скобки по бокам ядра делают герб программистским без прямого клише с клавиатурой или ноутбуком.",
  },
  {
    title: "Честные весы",
    text: "Внутренняя перекладина с симметричными чашами означает прямоту, внутреннюю этику и стремление держать слово даже под нагрузкой.",
  },
  {
    title: "Пульс тревожности",
    text: "Мягкое свечение и концентрические волны не романтизируют тревогу, а показывают её как чувствительный сенсор окружающего мира.",
  },
  {
    title: "Ленты танца",
    text: "Две коралловые ленты работают как геральдические держатели и одновременно как движение латиноамериканского танца: ритм, пластика, жизнь.",
  },
];

const facets = [
  "умный",
  "тревожный, но собранный",
  "честный",
  "программист",
  "любитель sci-fi",
  "любитель латинских танцев",
];

export default function CoatOfArmsPage() {
  return (
    <div className={`page ${styles.page}`}>
      <section className={styles.hero}>
        <div className={`${styles.panel} ${styles.crestPanel}`}>
          <svg
            className={styles.crestSvg}
            viewBox="0 0 560 760"
            role="img"
            aria-labelledby="crest-title crest-desc"
          >
            <title id="crest-title">Персональный герб программиста</title>
            <desc id="crest-desc">
              Тёмный щит со звёздной короной, орбитальным кольцом, кодовыми
              скобками, весами честности и танцующими лентами.
            </desc>

            <defs>
              <linearGradient id="shieldFill" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#16203d" />
                <stop offset="45%" stopColor="#0f1732" />
                <stop offset="100%" stopColor="#090d19" />
              </linearGradient>
              <linearGradient id="shieldEdge" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#f7f8ff" />
                <stop offset="35%" stopColor="#9eb4ff" />
                <stop offset="70%" stopColor="#7ae3ff" />
                <stop offset="100%" stopColor="#f7d28e" />
              </linearGradient>
              <linearGradient id="coreGlow" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#ffd892" />
                <stop offset="100%" stopColor="#ff8d63" />
              </linearGradient>
              <linearGradient id="ribbonFill" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#ff936b" />
                <stop offset="100%" stopColor="#ff4f86" />
              </linearGradient>
              <radialGradient id="halo" cx="50%" cy="45%" r="60%">
                <stop offset="0%" stopColor="rgba(153, 191, 255, 0.95)" />
                <stop offset="55%" stopColor="rgba(94, 127, 255, 0.35)" />
                <stop offset="100%" stopColor="rgba(94, 127, 255, 0)" />
              </radialGradient>
              <filter id="softGlow" x="-60%" y="-60%" width="220%" height="220%">
                <feGaussianBlur stdDeviation="12" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>

            <ellipse cx="280" cy="355" rx="170" ry="205" fill="url(#halo)" />

            <g className={styles.starsDrift} opacity="0.95">
              <path
                d="M184 86l6 13 14 2-10 10 3 14-13-7-12 7 2-14-10-10 14-2 6-13Z"
                fill="#d7e4ff"
              />
              <path
                d="M280 52l8 17 18 3-13 12 4 19-17-9-17 9 4-19-13-12 18-3 8-17Z"
                fill="#fff3c9"
              />
              <path
                d="M376 86l6 13 14 2-10 10 3 14-13-7-12 7 2-14-10-10 14-2 6-13Z"
                fill="#d7e4ff"
              />
            </g>

            <g className={styles.orbitSpin}>
              <ellipse
                cx="280"
                cy="176"
                rx="122"
                ry="38"
                fill="none"
                stroke="#90b4ff"
                strokeWidth="4"
                strokeDasharray="12 8"
                opacity="0.8"
              />
              <circle cx="161" cy="175" r="9" fill="#8ef0ff" />
              <circle cx="397" cy="177" r="9" fill="#ffd893" />
            </g>

            <g className={styles.danceLeft}>
              <path
                d="M85 248c-30 45-45 93-42 146 3 45 23 88 59 127 14 15 34 30 61 43-16-21-27-44-31-68-6-40 3-80 25-121 23-41 32-72 29-93-4-20-11-31-24-34-23-4-45-3-77 0Z"
                fill="url(#ribbonFill)"
                opacity="0.84"
              />
            </g>

            <g className={styles.danceRight}>
              <path
                d="M475 248c30 45 45 93 42 146-3 45-23 88-59 127-14 15-34 30-61 43 16-21 27-44 31-68 6-40-3-80-25-121-23-41-32-72-29-93 4-20 11-31 24-34 23-4 45-3 77 0Z"
                fill="url(#ribbonFill)"
                opacity="0.84"
              />
            </g>

            <path
              d="M280 126c99 0 178 55 178 126v115c0 126-81 211-178 267-97-56-178-141-178-267V252c0-71 79-126 178-126Z"
              fill="url(#shieldFill)"
              stroke="url(#shieldEdge)"
              strokeWidth="8"
            />

            <path
              d="M280 154c82 0 148 44 148 101v104c0 105-67 177-148 226-81-49-148-121-148-226V255c0-57 66-101 148-101Z"
              fill="none"
              stroke="rgba(255,255,255,0.16)"
              strokeWidth="2"
            />

            <g opacity="0.72">
              <path
                d="M154 279h252"
                stroke="rgba(147,175,255,0.36)"
                strokeWidth="2"
                strokeDasharray="6 10"
              />
              <path
                d="M174 332h212"
                stroke="rgba(147,175,255,0.22)"
                strokeWidth="2"
                strokeDasharray="6 10"
              />
              <path
                d="M198 386h164"
                stroke="rgba(147,175,255,0.18)"
                strokeWidth="2"
                strokeDasharray="6 10"
              />
            </g>

            <g className={styles.pulse} filter="url(#softGlow)">
              <circle
                cx="280"
                cy="318"
                r="68"
                fill="none"
                stroke="rgba(144, 180, 255, 0.18)"
                strokeWidth="10"
              />
              <circle
                cx="280"
                cy="318"
                r="38"
                fill="url(#coreGlow)"
                opacity="0.95"
              />
              <path
                d="M280 274l15 29 33 5-24 22 6 32-30-16-30 16 6-32-24-22 33-5 15-29Z"
                fill="#fff7d8"
              />
            </g>

            <g opacity="0.92">
              <path
                d="M192 262c-22 14-39 33-49 56 10 25 27 44 49 58"
                fill="none"
                stroke="#e8f0ff"
                strokeWidth="8"
                strokeLinecap="round"
              />
              <path
                d="M368 262c22 14 39 33 49 56-10 25-27 44-49 58"
                fill="none"
                stroke="#e8f0ff"
                strokeWidth="8"
                strokeLinecap="round"
              />
            </g>

            <g>
              <path
                d="M205 430h150"
                stroke="#f4d995"
                strokeWidth="7"
                strokeLinecap="round"
              />
              <path
                d="M280 430v95"
                stroke="#f4d995"
                strokeWidth="7"
                strokeLinecap="round"
              />
              <path
                d="M230 430c-10 20-23 33-39 39"
                stroke="#f4d995"
                strokeWidth="5"
                strokeLinecap="round"
                fill="none"
              />
              <path
                d="M330 430c10 20 23 33 39 39"
                stroke="#f4d995"
                strokeWidth="5"
                strokeLinecap="round"
                fill="none"
              />
              <path
                d="M160 470c9 19 22 29 39 29s30-10 39-29c-11-5-24-8-39-8s-28 3-39 8Z"
                fill="rgba(244, 217, 149, 0.22)"
                stroke="#f4d995"
                strokeWidth="4"
              />
              <path
                d="M322 470c9 19 22 29 39 29s30-10 39-29c-11-5-24-8-39-8s-28 3-39 8Z"
                fill="rgba(244, 217, 149, 0.22)"
                stroke="#f4d995"
                strokeWidth="4"
              />
            </g>

            <g opacity="0.9">
              <path
                d="M221 553h118"
                stroke="rgba(255,255,255,0.18)"
                strokeWidth="2"
                strokeDasharray="5 8"
              />
              <path
                d="M251 584h58"
                stroke="rgba(255,255,255,0.18)"
                strokeWidth="2"
                strokeDasharray="5 8"
              />
            </g>

            <g>
              <path
                d="M191 632c24 10 55 16 89 16s65-6 89-16"
                fill="none"
                stroke="#f0f3ff"
                strokeWidth="4"
                strokeLinecap="round"
              />
              <text
                x="280"
                y="680"
                textAnchor="middle"
                fill="#fff5d0"
                style={{
                  fontSize: 24,
                  letterSpacing: "0.28em",
                  fontWeight: 700,
                }}
              >
                CLARUS CODICE
              </text>
            </g>
          </svg>
        </div>

        <div className={`${styles.panel} ${styles.contentPanel}`}>
          <div className={styles.eyebrow}>CSS + SVG coat of arms</div>
          <h1 className={styles.title}>Герб спокойного ума с внутренним радаром</h1>
          <p className={styles.lead}>
            Я придумал тебе техно-геральдический герб: не рыцарский музейный
            реквизит, а личный знак человека, который думает глубоко, пишет код
            честно, любит космическое воображение и переводит напряжение в
            движение.
          </p>

          <div className={styles.statRow}>
            {facets.map((facet) => (
              <div key={facet} className={styles.pill}>
                {facet}
              </div>
            ))}
          </div>

          <div className={styles.story}>
            <div className={styles.storyTitle}>Краткое чтение герба</div>
            <ul className={styles.storyList}>
              <li className={styles.storyItem}>
                <strong>Цвета</strong>
                <span>
                  Ночной синий и серебро — интеллект, ясность и дисциплина;
                  янтарное ядро — живая мысль; коралл — тело, ритм и танец.
                </span>
              </li>
              <li className={styles.storyItem}>
                <strong>Характер</strong>
                <span>
                  Центр герба светится, но окружён кольцами наблюдения: ты не
                  хаотичен, ты внимателен и редко отключаешь внутренний сканер.
                </span>
              </li>
              <li className={styles.storyItem}>
                <strong>Мотто</strong>
                <span>
                  <em>Clarus Codice</em> — «ясен в коде». Это и про ремесло, и
                  про честность: не запутывать ни себя, ни других.
                </span>
              </li>
            </ul>
          </div>
        </div>
      </section>

      <section className={styles.legendGrid}>
        {legendItems.map((item) => (
          <article key={item.title} className={styles.legendCard}>
            <h2>{item.title}</h2>
            <p>{item.text}</p>
          </article>
        ))}
      </section>

      <p className={styles.note}>
        Если захочешь, следующий шаг можно сделать в одном из трёх направлений:
        более строгий геральдический стиль, более киберпанковая версия или
        вариант под аватар/стикер/печать.
      </p>
    </div>
  );
}
