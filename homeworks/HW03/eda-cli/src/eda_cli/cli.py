from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import typer

from .core import (
    DatasetSummary,
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
    find_outliers,
    get_problematic_columns,
)
from .viz import (
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_histograms_per_column,
    save_top_categories_tables,
)

app = typer.Typer(help="Мини-CLI для EDA CSV-файлов")


def _load_csv(
    path: Path,
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Загружает CSV файл в DataFrame.
    """
    if not path.exists():
        raise typer.BadParameter(f"Файл '{path}' не найден")
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except Exception as exc:
        raise typer.BadParameter(f"Не удалось прочитать CSV: {exc}") from exc


def _generate_report_content(
    df: pd.DataFrame,
    summary: DatasetSummary,
    missing_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    top_cats: Dict[str, pd.DataFrame],
    quality_flags: Dict[str, any],
    outliers_info: Dict[str, List[int]],
    problematic_columns: List[str],
    path: str,
    title: str,
    max_hist_columns: int,
    top_k_categories: int,
    min_missing_share: float,
    check_outliers: bool,
    sep: str,
    encoding: str,
) -> str:
    """
    Генерирует содержимое отчета в формате Markdown.
    """
    content = []
    
    # Заголовок
    content.append(f"# {title}\n")
    content.append(f"*Сгенерировано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    # Параметры анализа
    content.append("\n## Параметры анализа")
    content.append(f"- Исходный файл: `{Path(path).name}`")
    content.append(f"- Разделитель: `{sep}`")
    content.append(f"- Кодировка: `{encoding}`")
    content.append(f"- Максимум гистограмм: `{max_hist_columns}`")
    content.append(f"- Топ категорий: `{top_k_categories}`")
    content.append(f"- Порог пропусков: `{min_missing_share}`")
    content.append(f"- Проверка выбросов: `{'Да' if check_outliers else 'Нет'}`\n")
    
    # Основная информация
    content.append("## Основная информация")
    content.append(f"- Строк: **{summary.n_rows}**")
    content.append(f"- Колонок: **{summary.n_cols}**\n")
    
    # Качество данных
    content.append("## Качество данных")
    content.append(f"- Общий скор качества: **{quality_flags.get('quality_score', 0):.2f}**")
    
    # Базовые флаги качества
    for flag in ['too_few_rows', 'too_many_columns', 'too_many_missing']:
        if flag in quality_flags:
            content.append(f"- {flag.replace('_', ' ').title()}: **{quality_flags[flag]}**")
    
    # Новые эвристики (HW03)
    new_flags = [
        'has_constant_columns',
        'has_imbalanced_categories', 
        'has_high_cardinality_categoricals',
        'has_suspicious_id_duplicates',
        'has_many_zero_values'
    ]
    
    content.append("\n### Новые эвристики качества (HW03)")
    for flag in new_flags:
        if flag in quality_flags:
            value = quality_flags[flag]
            if isinstance(value, bool):
                value_str = "✅ Да" if value else "❌ Нет"
            else:
                value_str = str(value)
            content.append(f"- {flag.replace('_', ' ').title()}: {value_str}")
    
    # Проблемные колонки (по пропускам)
    if problematic_columns:
        content.append(f"\n## Проблемные колонки (пропусков > {min_missing_share*100:.1f}%)")
        for col in problematic_columns:
            missing_pct = missing_df.loc[col, 'missing_share'] * 100
            content.append(f"- `{col}`: {missing_pct:.1f}% пропусков")
    
    # Выбросы
    if outliers_info and check_outliers:
        content.append("\n## Выбросы")
        for col, indices in outliers_info.items():
            content.append(f"- `{col}`: {len(indices)} выбросов")
    
    # Пропуски
    content.append("\n## Пропуски")
    if missing_df.empty:
        content.append("Пропусков нет.")
    else:
        content.append("Таблица пропусков сохранена в `missing.csv`")
        content.append("Визуализация пропусков: `missing_matrix.png`\n")
    
    # Корреляция
    content.append("\n## Корреляция числовых признаков")
    if corr_df.empty:
        content.append("Недостаточно числовых колонок для корреляции.")
    else:
        content.append("Корреляционная матрица сохранена в `correlation.csv`")
        content.append("Тепловая карта корреляций: `correlation_heatmap.png`\n")
    
    # Категориальные признаки
    content.append("\n## Категориальные признаки")
    if not top_cats:
        content.append("Категориальные/строковые признаки не найдены.")
    else:
        content.append(f"Топ-{top_k_categories} категорий:")
        for col_name, cat_df in top_cats.items():
            content.append(f"\n### {col_name}")
            for _, row in cat_df.head(top_k_categories).iterrows():
                content.append(f"- `{row['value']}`: {int(row['count'])} ({row['share']*100:.1f}%)")
    
    # Визуализации
    content.append("\n## Визуализации")
    content.append(f"- Гистограммы числовых колонок (максимум {max_hist_columns}): `hist_*.png`")
    content.append("- Матрица пропусков: `missing_matrix.png`")
    if not corr_df.empty:
        content.append("- Тепловая карта корреляций: `correlation_heatmap.png`")
    
    return "\n".join(content)


@app.command()
def overview(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    """
    Напечатать краткий обзор датасета.
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    summary: DatasetSummary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)
    
    missing_df = missing_table(df)
    quality_flags = compute_quality_flags(summary, missing_df, df)

    typer.echo(f"Строк: {summary.n_rows}")
    typer.echo(f"Столбцов: {summary.n_cols}")
    typer.echo(f"Общий скор качества: {quality_flags.get('quality_score', 0):.2f}")
    
    # Выводим новые флаги качества
    new_flags = [
        'has_constant_columns',
        'has_imbalanced_categories', 
        'has_high_cardinality_categoricals',
        'has_suspicious_id_duplicates'
    ]
    
    typer.echo("\nФлаги качества:")
    for flag in new_flags:
        if flag in quality_flags:
            value = quality_flags[flag]
            typer.echo(f"  - {flag.replace('_', ' ').title()}: {value}")
    
    typer.echo("\nКолонки:")
    typer.echo(summary_df.to_string(index=False))


@app.command()
def report(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    out_dir: str = typer.Option("reports", help="Каталог для отчёта."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
    max_hist_columns: int = typer.Option(
        6, 
        help="Максимум числовых колонок для гистограмм."
    ),
    check_outliers: bool = typer.Option(
        False, 
        help="Проверять выбросы."
    ),
    verbose: bool = typer.Option(
        False, 
        help="Подробный вывод."
    ),
    top_k_categories: int = typer.Option(
        5, 
        help="Количество топ-категорий для вывода."
    ),
    title: str = typer.Option(
        "EDA Report", 
        help="Заголовок отчета."
    ),
    min_missing_share: float = typer.Option(
        0.3, 
        min=0.0, 
        max=1.0,
        help="Порог доли пропусков для проблемных колонок."
    ),
) -> None:
    """
    Сгенерировать полный EDA-отчёт с новыми параметрами для HW03.
    """
    # Создаем директорию для отчета
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Загружаем данные
    if verbose:
        typer.echo(f"Загрузка файла: {path}")
    
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    
    if verbose:
        typer.echo(f"  ✓ Загружено: {df.shape[0]} строк, {df.shape[1]} колонок")

    # 1. Вычисляем основные метрики
    summary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)
    missing_df = missing_table(df)
    corr_df = correlation_matrix(df)
    top_cats = top_categories(df, top_k=top_k_categories)
    
    if verbose:
        typer.echo("  ✓ Вычислены основные метрики")

    # 2. Вычисляем качество данных с новыми эвристиками
    quality_flags = compute_quality_flags(summary, missing_df, df)
    
    if verbose:
        typer.echo(f"  ✓ Вычислено качество данных: {quality_flags.get('quality_score', 0):.2f}")

    # 3. Находим выбросы (если включено)
    outliers_info = {}
    if check_outliers:
        outliers_info = find_outliers(df)
        if verbose and outliers_info:
            typer.echo(f"  ✓ Найдены выбросы в {len(outliers_info)} колонках")

    # 4. Определяем проблемные колонки по пропускам
    problematic_columns = get_problematic_columns(missing_df, min_missing_share)
    
    if verbose and problematic_columns:
        typer.echo(f"  ✓ Обнаружены проблемные колонки: {len(problematic_columns)}")

    # 5. Сохраняем табличные данные
    summary_df.to_csv(out_root / "summary.csv", index=False)
    
    if not missing_df.empty:
        missing_df.to_csv(out_root / "missing.csv", index=True)
    
    if not corr_df.empty:
        corr_df.to_csv(out_root / "correlation.csv", index=True)
    
    if top_cats:
        save_top_categories_tables(top_cats, out_root / "top_categories")

    if verbose:
        typer.echo("  ✓ Сохранены табличные файлы")

    # 6. Генерируем Markdown отчет
    md_content = _generate_report_content(
        df=df,
        summary=summary,
        missing_df=missing_df,
        corr_df=corr_df,
        top_cats=top_cats,
        quality_flags=quality_flags,
        outliers_info=outliers_info,
        problematic_columns=problematic_columns,
        path=path,
        title=title,
        max_hist_columns=max_hist_columns,
        top_k_categories=top_k_categories,
        min_missing_share=min_missing_share,
        check_outliers=check_outliers,
        sep=sep,
        encoding=encoding,
    )
    
    md_path = out_root / "report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(md_content)
    
    if verbose:
        typer.echo(f"  ✓ Создан отчет: {md_path}")

    # 7. Генерируем визуализации
    plot_histograms_per_column(df, out_root, max_columns=max_hist_columns)
    plot_missing_matrix(df, out_root / "missing_matrix.png")
    
    if not corr_df.empty:
        plot_correlation_heatmap(df, out_root / "correlation_heatmap.png")
    
    if verbose:
        typer.echo("  ✓ Созданы визуализации")

    # 8. Итоговое сообщение
    typer.echo(f"\n✅ Отчёт успешно сгенерирован в каталоге: {out_dir}")
    typer.echo(f"   - Основной отчет: {md_path}")
    
    if problematic_columns:
        typer.echo(f"   - Проблемные колонки ({len(problematic_columns)}): {', '.join(problematic_columns)}")
    
    if check_outliers and outliers_info:
        total_outliers = sum(len(indices) for indices in outliers_info.values())
        typer.echo(f"   - Выбросы: {total_outliers} в {len(outliers_info)} колонках")


if __name__ == "__main__":
    app()