from sqlalchemy import (
    create_engine,
    Table,
    MetaData,
    Engine,
    column,
    literal_column,
    text,
    select,
    extract,
    literal,
    null,
    cast,
    String,
    DOUBLE_PRECISION,
    NUMERIC,
    INTEGER,
    TEXT,
    and_,
    union_all,
)
from sqlalchemy.sql import values, func, alias, lateral, true
import os
from dataclasses import dataclass
from typing import Literal
from sqlalchemy.dialects import postgresql


@dataclass
class TableTokenizationSpec:
    table_name: str
    event_type: Literal["infusion", "onetime"]
    ignore_cols: list = None
    modulated_cols: dict = None
    needs_alignment: bool = False

    def __post_init__(self):
        if self.ignore_cols is None:
            self.ignore_cols = list()

        if self.modulated_cols is None:
            self.modulated_cols = dict()

        # columns we never want to tokenize
        self.ignore_cols += [
            "subject_id",
            "hadm_id",
            "stay_id",
            "specimen_id",
            "charttime",
            "starttime",
            "endtime",
        ]

    def get_numeric_columns(self, table: Table):
        return [
            i.name
            for i in table.c
            if (isinstance(i.type, DOUBLE_PRECISION) or isinstance(i.type, NUMERIC))
            and i.name not in self.ignore_cols
            and i.name not in self.modulated_cols.keys()
            and i.name not in self.modulated_cols.values()
        ]

    def get_categorical_columns(self, table: Table):
        return [
            i.name
            for i in table.c
            if (isinstance(i.type, TEXT) or isinstance(i.type, INTEGER))
            and i.name not in tts.ignore_cols
            and i.name not in tts.modulated_cols.keys()
            and i.name not in tts.modulated_cols.values()
        ]


TTSs = [
    TableTokenizationSpec(
        "vitalsign",
        "onetime",
        ["mbp", "sbp_ni", "dbp_ni", "mbp_ni"],
        {"temperature": "temperature_site"},
    ),
    TableTokenizationSpec("crrt", "onetime"),
    TableTokenizationSpec("norepinephrine_equivalent_dose", "infusion"),
    TableTokenizationSpec("chemistry", "onetime", ["aniongap"], needs_alignment=True),
]


def build_table_stmt_onetime(tts: TableTokenizationSpec, table: Table):

    numeric_cols = tts.get_numeric_columns(table)
    categorical_cols = tts.get_categorical_columns(table)

    tokenization_data_expr = [
        (literal(f"{tts.table_name}.{cname}"), table.c[cname], table.c[cname] == None)
        for cname in numeric_cols
    ]

    tokenization_data_expr += [
        (
            func.concat(
                literal(f"{tts.table_name}.{cname}."), cast(table.c[cname], TEXT)
            ),
            None,
            table.c[cname] == None,
        )
        for cname in categorical_cols
    ]

    tokenization_data_expr += [
        (
            func.concat(
                literal(f"{tts.table_name}.{cname}."), cast(table.c[mod_cname], String)
            ),
            table.c[cname],
            table.c[cname] == None,
        )
        for cname, mod_cname in tts.modulated_cols.items()
    ]

    tokens = values(
        column("token_label"), column("token_value"), column("token_null")
    ).data(tokenization_data_expr)
    tokens = lateral(tokens).alias("tokens")

    return (
        select(
            table.c.stay_id,
            table.c.charttime,
            tokens.c.token_label,
            tokens.c.token_value,
        )
        .select_from(table.join(tokens, true()))
        .where(~tokens.c.token_null)
    ).cte(f"{tts.table_name}_tokenized")


def build_table_stmt_infusion(ttd: TableTokenizationSpec, table: Table):
    numeric_cols = tts.get_numeric_columns(table)
    categorical_cols = tts.get_categorical_columns(table)

    assert len(categorical_cols) == 0, "Categorical infusion events not yet supported"
    assert len(ttd.modulated_cols) == 0, "Modulated infusion events not yet supported"

    tokenization_data_expr = list()

    for ncol in numeric_cols:
        # TODO: divide by time
        tokenization_data_expr.append(
            (
                table.c.starttime,
                literal(f"{tts.table_name}.rate"),
                table.c[ncol]
                / (extract("epoch", table.c.endtime - table.c.starttime) / 3600),
            )
        )

        tokenization_data_expr.append(
            (table.c.endtime, literal(f"{tts.table_name}.rate"), 0.0)
        )

    tokens = values(
        column("charttime"), column("token_label"), column("token_value")
    ).data(tokenization_data_expr)
    tokens = lateral(tokens).alias("tokens")

    return (
        select(
            table.c.stay_id,
            tokens.c.charttime,
            tokens.c.token_label,
            tokens.c.token_value,
        )
        .select_from(table.join(tokens, true()))
        .cte(f"{tts.table_name}_tokenized")
    )


def do_alignment(tts: TableTokenizationSpec, table: Table, icustays: Table):
    cte = (
        select(icustays.c.stay_id, *[i for i in table.columns])
        .select_from(
            table.join(
                icustays,
                and_(
                    table.c.hadm_id == icustays.c.hadm_id,
                    table.c.charttime >= icustays.c.icu_intime,
                    table.c.charttime <= icustays.c.icu_outtime,
                ),
            )
        )
        .where(icustays.c.stay_id != None)
    ).cte(f"{tts.table_name}_aligned")

    return cte


if __name__ == "__main__":
    user = os.environ.get("PGUSER", "postgres")
    password = os.environ.get("PGPASSWORD", "")
    host = os.environ.get("PGHOST", "localhost")
    port = os.environ.get("PGPORT", "5432")
    dbname = os.environ.get("PGDATABASE", "mimiciv")

    engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")

    metadata = MetaData()

    icustays = Table(
        "icustay_detail", metadata, autoload_with=engine, schema="mimiciv_derived"
    )

    ctes_for_union = list()

    for tts in TTSs:
        table = Table(
            tts.table_name, metadata, autoload_with=engine, schema="mimiciv_derived"
        )

        if tts.needs_alignment:
            alignment_cte = do_alignment(tts, table, icustays)

            # print()
            # print(
            #     alignment_cte.compile(
            #         dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
            #     )
            # )

            table = alignment_cte

        if tts.event_type == "onetime":
            ctes_for_union.append(build_table_stmt_onetime(tts, table))
        elif tts.event_type == "infusion":
            ctes_for_union.append(build_table_stmt_infusion(tts, table))

    stmt = union_all(
        *[
            select(cte.c.stay_id, cte.c.charttime, cte.c.token_label, cte.c.token_value)
            for cte in ctes_for_union
        ]
    ).order_by("stay_id", "charttime")

    print(
        stmt.compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    )
