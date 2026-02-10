# Исследование spark.js: поддержка SOG2 и LOD

## Что такое spark.js

**spark.js** (`@sparkjsdev/spark`) — это продвинутый рендерер 3D Gaussian Splatting для THREE.js.

- **Репозиторий**: https://github.com/sparkjsdev/spark
- **Сайт**: https://sparkjs.dev/
- **NPM**: `@sparkjsdev/spark` (текущая версия: 0.1.10)
- **Лицензия**: MIT

### Ключевые возможности (main ветка, v0.1.10)

- Интеграция с THREE.js рендер-пайплайном
- Поддержка WebGL2 (98%+ устройств)
- Оптимизация для мобильных устройств
- Поддержка множества форматов: `.PLY`, `.SPZ`, `.SPLAT`, `.KSPLAT`, `.SOG`
- Рендеринг множественных splat-объектов с корректной сортировкой
- Шейдерный граф для динамического создания/редактирования сплатов на GPU
- Скелетная анимация сплатов

## Поддержка SOG/SOG2

### Что такое SOG/SOG2

SOG (Scene Object Graph) — формат сжатия 3D Gaussian Splatting, разработанный PlayCanvas:
- **SOG v1** (SOGS): https://blog.playcanvas.com/playcanvas-adopts-sogs-for-20x-3dgs-compression/
- **SOG v2**: https://blog.playcanvas.com/playcanvas-open-sources-sog-format-for-gaussian-splatting/

SOG2 использует codebook-сжатие для scales и sh0 (вместо min/max интерполяции в v1), а также поддерживает поля `version: 2` и `count`.

### Текущий статус поддержки SOG2 в spark.js (main ветка)

**SOG2 полностью поддерживается в стабильной версии 0.1.10.**

#### История добавления:
| Версия | Дата | Что добавлено |
|--------|------|---------------|
| 0.1.5 | 1 июля 2025 | Базовая поддержка SOGS (SOG v1) |
| 0.1.6 | 11 июля 2025 | Загрузка SOGS из .zip файлов |
| 0.1.7 | 30 июля 2025 | Оптимизация параллельного декодирования SOGS |
| 0.1.9 | 22 сентября 2025 | Оптимизация parsing и lookup-таблиц для SOGS |
| **0.1.10** | **24 октября 2025** | **SOG v2 поддержка** |

#### Реализация SOG2 в коде:

1. **`src/SplatLoader.ts`** — определяет типы `PcSogsJson` (v1) и `PcSogsV2Json` (v2), автоматически определяет формат файла:
   - Расширение `.sog` маппится на `SplatFileType.PCSOGSZIP`
   - Поддерживает загрузку как отдельных JSON с WebP-файлами, так и ZIP-архивов
   - Функция `tryPcSogs()` парсит JSON и определяет v1/v2 по наличию поля `version`

2. **`src/pcsogs.ts`** — декодирование SOG данных:
   - Функция `unpackPcSogs()` обрабатывает как v1, так и v2
   - Для v2 используются codebook для scales, sh0
   - Поддержка SH-коэффициентов (sh1, sh2, sh3) для обеих версий
   - Декодирование WebP-изображений через offscreen WebGL2 контекст
   - Функция `unpackPcSogsZip()` для загрузки из ZIP-архивов

3. **Rust парсер** (`rust/spark-lib/src/sogs.rs` в splat-quick-lod ветке):
   - Полная реализация парсинга SOG v1 и v2 на Rust/WASM
   - Структуры `PcSogsV1` и `PcSogsV2` с serde десериализацией

### Пример загрузки SOG файла:

```javascript
import { SplatMesh } from "@sparkjsdev/spark";

// SOG v2 файл автоматически определяется по расширению .sog
const splat = new SplatMesh({ url: "https://example.com/scene.sog" });
scene.add(splat);

// Или ZIP с SOG данными
const splat2 = new SplatMesh({ url: "https://example.com/scene.zip" });
scene.add(splat2);
```

## Поддержка LOD (Level of Detail)

### Текущий статус LOD в main ветке

**LOD НЕ поддерживается в текущей стабильной версии (0.1.10 на main ветке).**

В main ветке нет `SplatPager.ts`, `NewSparkRenderer.ts`, RAD формата или LOD-деревьев.

### LOD в feature-ветках

LOD находится в активной разработке в нескольких ветках:

#### 1. Ветка `splat-quick-lod` (наиболее развитая)

Эта ветка содержит полную реализацию LOD-системы:

**Новые файлы TypeScript:**
- `src/SplatPager.ts` — основной класс для LOD-пагинации сплатов:
  - `PagedSplats` — источник сплатов с поддержкой LOD, управляет загрузкой чанков
  - `SplatPager` — менеджер страниц GPU-текстур для LOD-данных
  - Поддержка потоковой загрузки чанков (`fetchDecodeChunk`)
  - LRU-кэширование страниц текстур
  - Поддержка Range-запросов для частичной загрузки файлов
- `src/NewSparkRenderer.ts` — новый рендерер с встроенной LOD-поддержкой
- `src/SparkPortals.ts` — порталы для LOD-переходов
- `src/SparkXr.ts` — WebXR интеграция с LOD
- `src/ExtSplats.ts` — расширенный формат сплатов (float32 x/y/z)

**Новый формат RAD (Radiance):**
- `rust/spark-lib/src/rad.rs` — формат файла для LOD-данных:
  - Magic: `RAD0` (0x30444152)
  - Chunk Magic: `RADC` (0x43444152)
  - Различные кодировки: F32, F16, R8 для центров, RGB, scales, ориентаций
  - Поддержка SH-коэффициентов
  - Сжатие чанков через deflate

**Алгоритмы LOD (Rust/WASM):**
- `rust/spark-lib/src/quick_lod.rs` — быстрый алгоритм построения LOD-дерева:
  - Пространственное хеширование (grid-based)
  - Hierarchical LOD по feature_size
  - Мерж сплатов по уровням
- `rust/spark-lib/src/bhatt_lod.rs` — LOD через расстояние Бхаттачарья:
  - Мерж гауссиан через Bhattacharyya distance
  - Более точный мерж с учётом формы гауссиан
- `rust/spark-lib/src/tiny_lod.rs` — компактная LOD-реализация
- `rust/spark-lib/src/chunk_tree.rs` — дерево чанков для потоковой загрузки:
  - Morton-code упорядочивание
  - Пространственное разбиение сцены
- `rust/build-lod/` — CLI-инструмент для предварительного построения LOD

**SplatMesh LOD-опции (в splat-quick-lod):**
```typescript
const splat = new SplatMesh({
  url: "scene-lod-0.spz",
  lod: true,                  // Включить LOD
  nonLod: true,               // Сохранить оригинальные данные
  lodScale: 1.0,              // Масштаб LOD
  outsideFoveate: 1.0,        // Фовеация за пределами фрустума
  behindFoveate: 1.0,         // Фовеация за камерой
  coneFov: 0.0,               // Угол конуса фовеации
  coneFoveate: 1.0,           // Фовеация на краю конуса
  paged: true,                // Потоковая загрузка чанков
});
```

**Потоковая LOD-загрузка:**
```javascript
// Автоматически подгружает -lod-1.spz, -lod-2.spz, ...
const splat = new SplatMesh({ 
  url: "scene-lod-0.spz",
  paged: true,
});
```

#### 2. Ветка `feature/portal-lod-prefetch`

Расширение LOD с предзагрузкой для порталов — телепортация между LOD-сценами с предварительной загрузкой данных следующей сцены.

#### 3. Ветка `fix/pager-assignment-for-new-meshes`

Исправление назначения пагинатора для новых mesh-объектов в LOD-системе.

#### 4. Ветка `sogs`

Минимальная ветка — добавляет только пример загрузки SOG-файла (Sutro Tower в ZIP).

## Поддержка SOG2 LOD (комбинация)

### Вопрос: поддерживает ли spark.js загрузку SOG2 с LOD?

**Краткий ответ: Нет, напрямую SOG2 LOD не поддерживается.**

**Подробно:**

1. **SOG2 загрузка** — полностью поддерживается в стабильной версии (main, v0.1.10).

2. **LOD-система** — находится в разработке (ветка `splat-quick-lod`), но использует собственный формат `RAD` и `.spz` чанки для LOD-данных, а **не SOG2**.

3. **SOG2 формат сам по себе не содержит LOD-информации** — это формат сжатия данных (codebook + WebP-изображения), без иерархической структуры уровней детализации.

4. **LOD в spark.js работает по другому принципу:**
   - Файлы разбиваются на чанки (`-lod-0.spz`, `-lod-1.spz`, ...) или хранятся в едином RAD-файле
   - Каждый чанк содержит сплаты определённого уровня детализации
   - LOD-дерево строится на основе feature_size гауссиан
   - Чанки подгружаются по мере приближения камеры

5. **Теоретическая совместимость:** SOG2 мог бы быть использован как формат сжатия для отдельных LOD-чанков, но на данный момент LOD-система использует только SPZ и RAD форматы для чанков.

## Рекомендации для нашего проекта

На основе исследования, для нашего ROADMAP:

1. **"Добавить поддержку загрузки SOG2"** — можно реализовать, используя подход из `pcsogs.ts` spark.js:
   - Парсинг JSON meta.json из ZIP-архива
   - Декодирование WebP-изображений через WebGL2
   - Codebook-декодирование для v2

2. **"Потоковая загрузка SOG2 LOD tiles"** — формат SOG2 не предоставляет LOD из коробки. Варианты:
   - Использовать подход spark.js: LOD-дерево + чанки в SPZ/RAD формате
   - Реализовать собственную LOD-систему поверх SOG2 чанков
   - Использовать hierarchical SOG2 с разбиением сцены на тайлы разной детализации

3. **Рассмотреть интеграцию с spark.js:**
   - Библиотека MIT-лицензирована
   - Хорошо интегрируется с THREE.js (который мы используем через R3F)
   - NPM пакет: `npm install @sparkjsdev/spark`

## Структура веток репозитория sparkjsdev/spark

| Ветка | Описание |
|-------|----------|
| `main` | Стабильная версия 0.1.10. SOG v1/v2, SPZ, PLY, KSPLAT, SPLAT |
| `sogs` | Пример загрузки SOG (Sutro Tower ZIP) |
| `splat-quick-lod` | **Основная LOD-ветка**: SplatPager, RAD формат, quick_lod/bhatt_lod/tiny_lod, chunk_tree, NewSparkRenderer, SparkPortals, SparkXr |
| `feature/portal-lod-prefetch` | LOD с предзагрузкой для порталов |
| `fix/pager-assignment-for-new-meshes` | Фикс пагинатора для новых meshes |
| `feature/mobile-joystick` | Мобильный джойстик |
| `painter` | Рисование сплатов |
| `sparkportal` | Порталы между сценами |
