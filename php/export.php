<?php
/**
 * Boots the MyHordes kernel and dumps game data as JSON for the knowledge base.
 *
 * Usage: php export.php [myhordes_dir=/app] [out_dir=/export]
 *
 * Output:
 *   fixtures/<Chain>.json      merged data of every MyHordes\Plugins\Fixtures\* chain
 *   rules/raw.json             rules.yml (+ rules_local), as loaded by LegacyRulesConfigurationProvider
 *   rules/resolved/<type>.json effective town rules per town type (as ConfMaster::getTownConfigurationByType)
 *   settings/<Enum>.json       schema of TownSetting / CitizenProperties (key, default, parent)
 *   manifest.json              git sha, timestamps, per-chain status
 */
declare(strict_types=1);

use App\Enum\Configuration\CitizenProperties;
use App\Enum\Configuration\Configuration;
use App\Enum\Configuration\TownSetting;
use App\Kernel;
use App\Structures\TownConf;
use Symfony\Component\Dotenv\Dotenv;

$app = rtrim($argv[1] ?? '/app', '/');
$out = rtrim($argv[2] ?? '/export', '/');

require "$app/vendor/autoload.php";
(new Dotenv())->bootEnv("$app/.env");

$kernel = new Kernel($_SERVER['APP_ENV'], (bool)$_SERVER['APP_DEBUG']);
$kernel->boot();
$container = $kernel->getContainer();

function normalize(mixed $v): mixed {
    if (is_array($v)) return array_map('normalize', $v);
    if ($v instanceof BackedEnum) return $v->value;
    if ($v instanceof UnitEnum) return $v->name;
    if ($v instanceof JsonSerializable) return normalize($v->jsonSerialize());
    if ($v instanceof DateTimeInterface) return $v->format(DATE_ATOM);
    if (is_object($v)) {
        // Cast exposes private/protected props as "\0Class\0prop" / "\0*\0prop"
        $props = ['__class' => $v::class];
        foreach ((array)$v as $k => $val)
            $props[preg_replace('/^\0[^\0]+\0/', '', (string)$k)] = normalize($val);
        return $props;
    }
    return $v;
}

function write_json(string $path, mixed $data): void {
    if (!is_dir(dirname($path))) mkdir(dirname($path), 0777, true);
    file_put_contents($path, json_encode(
        normalize($data),
        JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES | JSON_PRESERVE_ZERO_FRACTION | JSON_THROW_ON_ERROR
    ));
}

function settings_schema(string $enum): array {
    /** @var Configuration $enum */
    $schema = [];
    foreach ($enum::cases() as $case) {
        try { $default = $case->default(); } catch (Throwable) { $default = null; }
        $schema[$case->name] = [
            'key'      => $case->key(),
            'abstract' => $case->abstract(),
            'parent'   => $case->parent()?->name,
            'default'  => $default,
            'translation_key' => $case->translationKey(),
        ];
    }
    return $schema;
}

$manifest = [
    'exported_at' => date(DATE_ATOM),
    'git_sha'     => trim((string)@shell_exec("git -C " . escapeshellarg($app) . " rev-parse HEAD 2>/dev/null")),
    'app_env'     => $_SERVER['APP_ENV'],
    'chains'      => [],
];

// Fixture chains (all public services, see myhordes-plugins services.yaml)
foreach (glob("$app/packages/myhordes-plugins/src/Fixtures/*.php") as $file) {
    $name = basename($file, '.php');
    $class = "MyHordes\\Plugins\\Fixtures\\$name";
    try {
        $data = $container->get($class)->data();
        write_json("$out/fixtures/$name.json", $data);
        $manifest['chains'][$name] = ['ok' => true, 'entries' => count($data)];
    } catch (Throwable $e) {
        $manifest['chains'][$name] = ['ok' => false, 'error' => $e->getMessage()];
    }
}

// Town rules, mirroring LegacyRulesConfigurationProvider + ConfMaster::getTownConfigurationByType
$rules = array_replace_recursive(
    $container->getParameter('rules'),
    $container->hasParameter('rules_local') ? $container->getParameter('rules_local') : []
);
write_json("$out/rules/raw.json", $rules);

$town_types = array_unique(array_merge(
    array_keys($rules),
    array_keys(json_decode(@file_get_contents("$out/fixtures/Town.json") ?: '{}', true) ?? [])
));
foreach ($town_types as $type) {
    $conf = new TownConf([$rules['default'] ?? [], $type === 'default' ? [] : ($rules[$type] ?? [])]);
    write_json("$out/rules/resolved/$type.json", $conf->getData());
    $private = new TownConf([$rules['default'] ?? [], $type === 'default' ? [] : ($rules[$type] ?? []), $rules['custom'] ?? []]);
    write_json("$out/rules/resolved/$type.private.json", $private->getData());
}
$manifest['town_types'] = array_values($town_types);

write_json("$out/settings/TownSetting.json", settings_schema(TownSetting::class));
write_json("$out/settings/CitizenProperties.json", settings_schema(CitizenProperties::class));

write_json("$out/manifest.json", $manifest);

$failed = array_keys(array_filter($manifest['chains'], fn($c) => !$c['ok']));
fwrite(STDOUT, sprintf("Exported %d chains (%d failed%s), %d town types to %s\n",
    count($manifest['chains']), count($failed), $failed ? ': ' . implode(', ', $failed) : '',
    count($town_types), $out));
exit($failed ? 1 : 0);
