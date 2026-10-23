-- MyHordes translations (German source text -> official text per language), so the bot can give the
-- official wording of game texts it finds in the code without a MyHordes checkout
CREATE TABLE translations (
    lang TEXT NOT NULL,
    de   TEXT NOT NULL,                         -- German source text, whitespace-normalized
    text TEXT NOT NULL
);
CREATE INDEX translations_lookup ON translations (lang, md5(de));
