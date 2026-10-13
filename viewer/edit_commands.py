
from __future__ import annotations
import math
import re
import unicodedata
from dataclasses import dataclass
from types import MappingProxyType

class EditCommandError(ValueError):
    pass

@dataclass(frozen=True, slots=True)
class EditCommandDefinition:
    command_id: str
    name: str
    aliases: tuple[str, ...]
    description: str
    optional_args: tuple[str, ...] = ()
    argument_unit: str = ''
    positive_arguments: bool = False
    help: str = ''
    argument_counts: tuple[int, ...] = ()

    @property
    def usage(self) -> str:
        arguments = ' [' + ' '.join(self.optional_args) + ']' if self.optional_args else ''
        return self.aliases[0] + arguments

    @property
    def accepted_counts(self) -> tuple[int, ...]:
        if self.argument_counts:
            return tuple(sorted({0, *self.argument_counts}))
        return (0, len(self.optional_args)) if self.optional_args else (0,)

@dataclass(frozen=True, slots=True)
class ParsedEditCommand:
    definition: EditCommandDefinition
    arguments: tuple[float, ...]
    entered_alias: str

    @property
    def command_id(self) -> str:
        return self.definition.command_id

    @property
    def name(self) -> str:
        return self.definition.name

    @property
    def has_arguments(self) -> bool:
        return bool(self.arguments)
EDIT_COMMANDS = (EditCommandDefinition('ALIGN:MOVE', 'MOVE', ('M', 'MOVE', 'TASI'), 'Secimi tasi', ('dx', 'dy', 'dz'), 'metre', help='M tasima aracini acar. M 1,5 -2 0 secimi goreli metre degerleriyle tasir.'), EditCommandDefinition('ALIGN:ROTATE Z', 'ROTATE', ('RO', 'R', 'ROTATE', 'DONDUR'), 'Secimi Z ekseninde dondur', ('aci',), 'derece', help='RO dondurme aracini acar. RO -90 secime goreli -90 derece donus uygular.'), EditCommandDefinition('ALIGN:SELECT', 'SELECT', ('SELECT', 'SEC'), 'Secim aracini ac', help='SELECT veya SEC secim aracini acar; sayisal deger almaz.'), EditCommandDefinition('REVIEW:HIDE', 'HIDE', ('H', 'HIDE', 'GIZLE'), 'Secilen taramalari gizle', help='H secilen taramalari gizler; kaynak veri silinmez.'), EditCommandDefinition('REVIEW:ISOLATE', 'ISOLATE', ('ISOLATE', 'YALIT'), 'Yalniz secilenleri goster', help='ISOLATE secilenleri gosterir, digerlerini gizler; kaynaklar korunur.'), EditCommandDefinition('REVIEW:SHOW ALL', 'SHOWALL', ('SHOWALL', 'HEPSI'), 'Tum taramalari goster', help='SHOWALL veya HEPSI taramalarin gorunurlugunu acar.'), EditCommandDefinition('REVIEW:ZOOM EXTENTS', 'EXTENTS', ('ZE', 'EXTENTS', 'TUMUNU'), 'Tum gorunur geometriyi kadraja al', help='ZE gorunur geometrinin tamamini kameraya sigdirir; tarama konumlari degismez.'), EditCommandDefinition('REVIEW:FIT SELECTED', 'FIT SELECTED', ('ZS', 'F', 'SECILI'), 'Secilenleri kadraja al', help='ZS veya F secilen gorunur taramalara odaklanir; tarama konumlari degismez.'), EditCommandDefinition('EDIT:UNDO', 'UNDO', ('UNDO', 'U', 'GERI'), 'Son duzenleme islemini geri al', help='UNDO mevcut geri alma kaydini uygular; geri alinabilir islem yoksa arayuz bilgi verir.'), EditCommandDefinition('ALIGN:LOCK', 'LOCK', ('LOCK', 'KILIT'), 'Secimin yerlesim kilidini degistir', help='LOCK veya KILIT secilen taramalarin yerlesim kilidini degistirir.'), EditCommandDefinition('ALIGN:PLAN PLACE', 'PLAN', ('PLAN', 'YERLESTIR'), 'Plan uzerinde yerlestirmeyi baslat', help='PLAN plan yerlestirme akisina girer; gerekli plan ve secim arayuzde denetlenir.'), EditCommandDefinition('ALIGN:AUTO ALIGN', 'AUTO', ('AUTO', 'OTOHIZALA'), 'Otomatik hizalamayi baslat', help='AUTO secilen taramalar icin otomatik hizalama akisini baslatir.'), EditCommandDefinition('REVIEW:ORBIT', 'ORBIT', ('OR', 'ORBIT', 'KAMERA'), 'Kamerayi dondur (sol tus orbit modu / OR daz del)', ('daz', 'del'), 'derece', help='OR orbit modunu acar veya kapatir: acikken sol tus kamerayi dondurur, Esc onceki araca doner. OR 30 kamerayi 30 derece azimut dondurur; OR -15 10 azimut ve yukselisi birlikte uygular. Ust gorunumde orbit kapalidir.', argument_counts=(1, 2)), EditCommandDefinition('REVIEW:CCA DENSITY', 'CCA', ('CCA',), 'Temsil yogunlugunu ayarla', ('yogunluk',), 'cca/m', True, 'CCA mevcut temsil yogunlugunu acar. CCA 250 pozitif 250 cca/m degerini ister.'))
COMMANDS_BY_ALIAS = MappingProxyType({alias: command for command in EDIT_COMMANDS for alias in command.aliases})
COMMANDS_BY_ID = MappingProxyType({command.command_id: command for command in EDIT_COMMANDS})
_DOT_NUMBER = re.compile('[+-]?(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+)(?:[eE][+-]?[0-9]+)?\\Z')
_COMMA_NUMBER = re.compile('[+-]?(?:[0-9]+,[0-9]+|,[0-9]+)(?:[eE][+-]?[0-9]+)?\\Z')

def _normalize_alias(alias: str) -> str:
    folded = unicodedata.normalize('NFKD', alias.casefold().replace('i', 'i'))
    return folded.encode('ascii', 'ignore').decode('ascii').upper()

def _read_number(token: str, command: EditCommandDefinition, index: int) -> float:
    valid = _COMMA_NUMBER.fullmatch(token) if ',' in token else _DOT_NUMBER.fullmatch(token)
    if valid is None:
        raise EditCommandError(f'{command.name}: {command.optional_args[index]} sonlu bir sayi olmali. Degerleri boslukla ayirin; ondalik icin 1.5 veya 1,5 yazin.')
    value = float(token.replace(',', '.'))
    if not math.isfinite(value):
        raise EditCommandError(f'{command.name}: {command.optional_args[index]} sonlu bir sayi olmali.')
    if command.positive_arguments and value <= 0:
        raise EditCommandError(f'{command.name}: {command.optional_args[index]} sifirdan buyuk olmali ({command.argument_unit}).')
    return value

def parse_edit_command(text: str) -> ParsedEditCommand:
    if not isinstance(text, str) or not text.strip():
        raise EditCommandError('Bir komut yazin. Ornek: M, RO, H, ZE veya CCA.')
    tokens = text.split()
    alias = _normalize_alias(tokens[0])
    command = COMMANDS_BY_ALIAS.get(alias)
    if command is None:
        raise EditCommandError('Komut taninmadi. Ornek: M, RO, H, ZE veya CCA. Komut listesinden bir ad secin.')
    argument_tokens = tokens[1:]
    if argument_tokens and (not command.optional_args):
        raise EditCommandError(f'{command.name} deger almaz. Kullanim: {command.usage}')
    if len(argument_tokens) not in command.accepted_counts:
        counts = ' veya '.join((str(count) for count in command.accepted_counts if count))
        raise EditCommandError(f'{command.name}: bos birakin veya {counts} deger girin ({command.argument_unit}). Kullanim: {command.usage}. Degerleri boslukla ayirin.')
    values = tuple((_read_number(token, command, index) for index, token in enumerate(argument_tokens)))
    return ParsedEditCommand(command, values, alias)

def command_completions(prefix: str='') -> tuple[str, ...]:
    if not isinstance(prefix, str):
        return ()
    prefix = prefix.lstrip()
    if any((character.isspace() for character in prefix)):
        return ()
    normalized = _normalize_alias(prefix)
    return tuple(sorted((alias for alias in COMMANDS_BY_ALIAS if alias.startswith(normalized))))

def command_help(alias: str | None=None) -> str:
    if alias is None or not alias.strip():
        return '\n'.join((f'{command.usage} - {command.description}' for command in EDIT_COMMANDS))
    command = COMMANDS_BY_ALIAS.get(_normalize_alias(alias.strip()))
    if command is None:
        raise EditCommandError('Bu komut icin yardim bulunamadi. Komut listesinden bir ad secin.')
    return f"{command.usage} - {command.description}\n{command.help}\nKisaltmalar: {', '.join(command.aliases)}"
