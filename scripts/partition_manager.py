#!/usr/bin/env python3
#
# Copyright (c) 2019 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-BSD-5-Clause-Nordic

import argparse
import yaml
import re
from os import path
import sys


def remove_item_not_in_list(list_to_remove_from, list_to_check):
    for x in list_to_remove_from:
        if x not in list_to_check and x != 'app':
            list_to_remove_from.remove(x)


def item_is_placed(d, item, after_or_before):
    assert(after_or_before in ['after', 'before'])
    return type(d['placement']) == dict and after_or_before in d['placement'].keys() and \
           d['placement'][after_or_before][0] == item


def remove_irrelevant_requirements(reqs):
    # Remove items dependencies to partitions which are not present
    for x in reqs.keys():
        if 'inside' in reqs[x].keys():
            [remove_item_not_in_list(reqs[x]['inside'], reqs.keys())]
            if not reqs[x]['inside']:
                del reqs[x]['inside']
        if 'share_size' in reqs[x].keys():
            [remove_item_not_in_list(reqs[x]['share_size'], reqs.keys())]
            if not reqs[x]['share_size']:
                del reqs[x]['share_size']
        for before_after in ['before', 'after']:
            if 'placement' in reqs[x] and type(reqs[x]['placement']) == dict and before_after in reqs[x]['placement'].keys():
                [remove_item_not_in_list(reqs[x]['placement'][before_after], reqs.keys())]
                if not reqs[x]['placement'][before_after]:
                    del reqs[x]['placement'][before_after]


def outermost(reqs, elem):
    comparison = 0
    while 'inside' in elem.keys() and elem['inside'][0] in reqs.keys():
        comparison += 1
        elem = reqs[elem['inside'][0]]
    return comparison


def get_images_which_need_resolving(reqs, sub_partitions):
    # Get candidates which have placement specs.
    unsorted = {x for x in reqs.keys() if 'placement' in reqs[x] and type(reqs[x]['placement']) == dict
                 and ('before' in reqs[x]['placement'].keys() or 'after' in reqs[x]['placement'].keys())}

    # Sort sub_partitions by whether they are inside other sub_partitions. Innermost first.
    sorted_subs = sorted(sub_partitions.values(), key=lambda x: outermost(sub_partitions, x), reverse=True)

    # Sort candidates by whether they are part of a sub_partitions.
    # sub_partition parts come last in the result list so they are more likely
    # to end up being placed next to each other, since they are inserted last.
    result = []
    for sub in sorted_subs:
        result = [part for part in sub['span'] if part in unsorted and part not in result] + result

    # Lastly, place non-partitioned parts at the front.
    result = [part for part in unsorted if part not in result] + result

    return result


def solve_direction(reqs, sub_partitions, unsolved, solution, ab):
    assert(ab in ['after', 'before'])
    current_index = 0
    pool = solution + list(sub_partitions.keys())
    current = pool[current_index]
    while current:
        depends = [x for x in unsolved if item_is_placed(reqs[x], current, ab)]
        if depends:
            # Place based on current, or based on the first/last element in the span of current.
            if ab == 'before':
                anchor = current if current in solution else next(solved for solved in solution
                                                                  if solved in sub_partitions[current]['span'])
                solution.insert(solution.index(anchor), depends[0])
            else:
                anchor = current if current in solution else next(solved for solved in reversed(solution)
                                                                  if solved in sub_partitions[current]['span'])
                solution.insert(solution.index(anchor) + 1, depends[0])
            unsolved.remove(depends[0])
            current = depends[0]
        else:
            current_index += 1
            if current_index >= len(pool):
                break
            current = pool[current_index]


def solve_first_last(reqs, unsolved, solution):
    for fl in ['first', 'last']:
        first_or_last = [x for x in reqs.keys() if 'placement' in reqs[x] and type(reqs[x]['placement']) == str and
                         reqs[x]['placement'] == fl]
        if first_or_last:
            assert(len(first_or_last) == 1)
            solution.insert(0 if fl is 'first' else len(solution), first_or_last[0])
            if first_or_last[0] in unsolved:
                unsolved.remove(first_or_last[0])


def extract_sub_partitions(reqs):
    sub_partitions = dict()
    keys_to_delete = list()

    for key, value in reqs.items():
        if 'inside' in value.keys():
            reqs[value['inside'][0]]['span'].append(key)
        if 'span' in value.keys():
            sub_partitions[key] = value
            keys_to_delete.append(key)

    # "Flatten" by changing all span lists to include the parts of partitions
    # instead of the partitions themselves.
    done = False
    while not done:
        done = True
        for key, value in reqs.items():
            if 'span' in value.keys():
                for part in value['span']:
                    if 'span' in reqs[part].keys():
                        value['span'].extend(reqs[part]['span'])
                        value['span'].remove(part)
                        value['span'] = list(set(value['span']))  # remove duplicates
                        done = False

    for key in keys_to_delete:
        del reqs[key]

    return sub_partitions


def resolve(reqs):
    solution = list(['app'])
    remove_irrelevant_requirements(reqs)
    sub_partitions = extract_sub_partitions(reqs)
    unsolved = get_images_which_need_resolving(reqs, sub_partitions)

    solve_first_last(reqs, unsolved, solution)
    while unsolved:
        solve_direction(reqs, sub_partitions, unsolved, solution, 'before')
        solve_direction(reqs, sub_partitions, unsolved, solution, 'after')

    # Validate partition spanning.
    for sub in sub_partitions:
        indices = [solution.index(part) for part in sub_partitions[sub]['span']]
        assert ((not indices) or (max(indices) - min(indices) + 1 == len(indices))), \
            "partition %s (%s) does not span over consecutive parts. Solution: %s" % \
            (sub, str(sub_partitions[sub]['span']), str(solution))
        assert all(part in solution for part in sub_partitions[sub]['span']), \
            "Some or all parts of partition %s have not been placed."

    return solution, sub_partitions


def find_partition_size_from_autoconf_h(configs):
    result = dict()
    for config in configs:
        with open(config, 'r') as cf:
            for line in cf.readlines():
                match = re.match(r'#define CONFIG_PM_PARTITION_SIZE_(\w*) (0x[0-9a-fA-F]*)', line)
                if match:
                    if int(match.group(2), 16) != 0:
                        result[match.group(1).lower()] = int(match.group(2), 16)

    return result


def get_list_of_autoconf_files(adr_map):
    return [path.join(props['out_dir'], 'autoconf.h') for props in adr_map.values() if 'out_dir' in props.keys()]


def load_size_config(adr_map):
    configs = get_list_of_autoconf_files(adr_map)
    size_configs = find_partition_size_from_autoconf_h(configs)

    for k, v in adr_map.items():
        if 'size' not in v.keys() and 'span' not in v.keys() and k != 'app' and 'share_size' not in v.keys():
            adr_map[k]['size'] = size_configs[k]


def load_adr_map(adr_map, input_files):
    for f in input_files:
        img_conf = yaml.safe_load(f)

        adr_map.update(img_conf)
    adr_map['app'] = dict()
    adr_map['app']['placement'] = ''


def app_size(reqs, total_size):
    size = total_size - sum([req['size'] for name, req in reqs.items() if 'size' in req.keys() and name is not 'app'])
    return size


def set_addresses(reqs, sub_partitions, solution, flash_size):
    set_shared_size(reqs, sub_partitions, flash_size)

    reqs['app']['size'] = app_size(reqs, flash_size)

    # First image starts at 0
    reqs[solution[0]]['address'] = 0
    for i in range(1, len(solution)):
        current = solution[i]
        previous = solution[i - 1]
        reqs[current]['address'] = reqs[previous]['address'] + reqs[previous]['size']


def set_size_addr(entry, size, address):
    entry['size'] = size
    entry['address'] = address


def set_sub_partition_address_and_size(reqs, sub_partitions):
    for sp_name, sp_value in sub_partitions.items():
        size = sum([reqs[part]['size'] for part in sp_value['span']])
        if size == 0:
            raise RuntimeError("No compatible parent partition found for %s" % sp_name)
        address = min([reqs[part]['address'] for part in sp_value['span']])
        if 'sub_partitions' in sp_value.keys():
            size = size // len(sp_value['sub_partitions'])
            for sub_partition in sp_value['sub_partitions']:
                sp_key_name = "%s_%s" % (sp_name, sub_partition)
                reqs[sp_key_name] = dict()
                set_size_addr(reqs[sp_key_name], size, address)
                address += size
        else:
            reqs[sp_name] = sp_value
            set_size_addr(reqs[sp_name], size, address)


def sizeof(reqs, req, total_size):
    if req == 'app':
        size = app_size(reqs, total_size)
    elif 'span' not in reqs[req].keys():
        size = reqs[req]['size'] if 'size' in reqs[req].keys() else 0
    else:
        size = sum([sizeof(reqs, part, total_size) for part in reqs[req]['span']])

    return size


def shared_size(reqs, share_with, total_size):
    sharer_count = reqs[share_with]['sharers']
    size = sizeof(reqs, share_with, total_size)
    if share_with == 'app' or ('span' in reqs[share_with] and 'app' in reqs[share_with]['span']):
        size /= (sharer_count + 1)
    return int(size)


def get_size_source(reqs, sharer):
    size_source = sharer
    while 'share_size' in reqs[size_source].keys():
        # Find "original" source.
        size_source = reqs[size_source]['share_size'][0]
    return size_source


def set_shared_size(reqs, sub_partitions, total_size):
    all_reqs = dict(reqs, **sub_partitions)
    for req in all_reqs:
        if 'share_size' in all_reqs[req].keys():
            size_source = get_size_source(all_reqs, req)
            if 'sharers' not in all_reqs[size_source].keys():
                all_reqs[size_source]['sharers'] = 0
            all_reqs[size_source]['sharers'] += 1
            all_reqs[req]['share_size'] = [size_source]
    new_sizes = dict()
    for req in all_reqs:
        if 'share_size' in all_reqs[req].keys():
            new_sizes[req] = shared_size(all_reqs, all_reqs[req]['share_size'][0], total_size)
    # Update all sizes after-the-fact or else the calculation will be messed up.
    for key, value in new_sizes.items():
        all_reqs[key]['size'] = value


def get_config(adr_map, config):
    config_file = get_list_of_autoconf_files(adr_map)[0]
    with open(config_file, "r") as cf:
        for line in cf.readlines():
            match = re.match(r'#define %s (\d*)' % config, line)
            if match:
                return int(match.group(1)) * 1024
        raise RuntimeError("Unable to find '%s' in any of: %s" % (config, config_file))


def get_flash_size(adr_map):
    return get_config(adr_map, "CONFIG_FLASH_SIZE")


def get_input_files(input_config):
    input_files = list()
    for v in input_config.values():
        input_files.append(open(v['pm.yml'], 'r'))
    return input_files


def add_configurations(adr_map, input_config):
    for k, v in input_config.items():
        adr_map[k]['out_dir'] = v['out_dir']
        adr_map[k]['build_dir'] = v['build_dir']


# def print_sub_region(region, size, reqs, solution):
def print_region(region, size, reqs, solution):
    print("%s (0x%x):" % (region, size))

    # Sort partitions three times:
    #  1. On whether they are a container (has a 'span'), containers first.
    #  2. On size, descending.
    #  3. On address, ascending.
    sorted_reqs = sorted(sorted(sorted(reqs.keys(), key=lambda x: int('span' in reqs[x]), reverse=True), key=lambda x: reqs[x]['size'], reverse=True), key=lambda x: reqs[x]['address'])
    sizes = ["%s%s (0x%x)" % (("| 0x%x: " % reqs[part]['address']) if 'span' not in reqs[part] else "+---", part, reqs[part]['size']) for part in sorted_reqs]
    maxlen = max(map(len, sizes)) + 1
    print("+%s+" % ("-"*(maxlen-1)))
    list(map(lambda s: print('%s' % s.ljust(maxlen, " ") + '|' if s[0] != '+' else s.ljust(maxlen, "-") + '+'), sizes))
    print("+%s+" % ("-"*(maxlen-1)))


def get_pm_config(input_config):
    adr_map = dict()
    input_files = get_input_files(input_config)
    load_adr_map(adr_map, input_files)
    add_configurations(adr_map, input_config)
    load_size_config(adr_map)
    flash_size = get_flash_size(adr_map)
    solution, sub_partitions = resolve(adr_map)
    set_addresses(adr_map, sub_partitions, solution, flash_size)
    set_sub_partition_address_and_size(adr_map, sub_partitions)
    print_region("FLASH", flash_size, adr_map, solution)
    return adr_map


def get_header_guard_start(filename):
    macro_name = filename.split('.h')[0]
    return '''/* File generated by %s, do not modify */
#ifndef %s_H__
#define %s_H__''' % (__file__, macro_name.upper(), macro_name.upper())


def get_header_guard_end(filename):
    return "#endif /* %s_H__ */" % filename.split('.h')[0].upper()


def get_config_lines(adr_map, head, split):
    lines = list()

    def fn(a, b):
        return lines.append(head + "PM_" + a + split + b)

    for area_name, area_props in sorted(adr_map.items(), key=lambda key_value: key_value[1]['address']):
        fn("%s_ADDRESS" % area_name.upper(), "0x%x" % area_props['address'])
        fn("%s_SIZE" % area_name.upper(), "0x%x" % area_props['size'])
        fn("%s_DEV_NAME" % area_name.upper(), "\"NRF_FLASH_DRV_NAME\"")

    flash_area_id = 0
    for area_name, area_props in sorted(adr_map.items(), key=lambda key_value: key_value[1]['address']):
        fn("%d_LABEL" % flash_area_id, "%s" % area_name.upper())
        adr_map[area_name]['flash_area_id'] = flash_area_id
        flash_area_id += 1

    for area_name, area_props in sorted(adr_map.items(), key=lambda key_value: key_value[1]['flash_area_id']):
        fn("%s_ID" % area_name.upper(), "%d" % area_props['flash_area_id'])
    fn("NUM", "%d" % flash_area_id)

    return lines


def only_sub_image_is_being_built(adr_map, app_output_dir):
    if len(adr_map) == 2:
        non_app_key = [non_app_key for non_app_key in adr_map.keys() if non_app_key != 'app'][0]
        return app_output_dir == adr_map[non_app_key]['out_dir']
    return False


def write_pm_config(adr_map, app_output_dir):
    pm_config_file = "pm_config.h"
    config_lines = get_config_lines(adr_map, "#define ", " ")

    for _, area_props in adr_map.items():
        area_props['pm_config'] = list.copy(config_lines)
        area_props['pm_config'].append("#define PM_ADDRESS 0x%x" % area_props['address'])
        area_props['pm_config'].append("#define PM_SIZE 0x%x" % area_props['size'])
        area_props['pm_config'].insert(0, get_header_guard_start(pm_config_file))
        area_props['pm_config'].append(get_header_guard_end(pm_config_file))

    # Store complete size/address configuration to all input paths
    for _, area_props in adr_map.items():
        if 'out_dir' in area_props.keys():
            write_pm_config_to_file(path.join(area_props['out_dir'], pm_config_file), area_props['pm_config'])

    # Store to root app, but
    if not only_sub_image_is_being_built(adr_map, app_output_dir):
        write_pm_config_to_file(path.join(app_output_dir, pm_config_file), adr_map['app']['pm_config'])


def write_pm_config_to_file(pm_config_file_path, pm_config):
    with open(pm_config_file_path, 'w') as out_file:
        out_file.write('\n'.join(pm_config))


def write_kconfig_file(adr_map, app_output_dir):
    pm_kconfig_file = "pm.config"
    config_lines = get_config_lines(adr_map, "", "=")

    # Store out dir and build dir
    for area_name, area_props in adr_map.items():
        if 'out_dir' in area_props.keys():
            config_lines.append('PM_%s_OUT_DIR="%s"' % (area_name.upper(), area_props['out_dir']))
            config_lines.append('PM_%s_BUILD_DIR="%s"' % (area_name.upper(), area_props['build_dir']))

    # Store output dir to app
    config_lines.append('PM_APP_OUT_DIR "%s"' % app_output_dir)

    # Store complete size/address configuration to all input paths
    for _, area_props in adr_map.items():
        if 'out_dir' in area_props.keys():
            write_pm_config_to_file(path.join(area_props['out_dir'], pm_kconfig_file), config_lines)

    # Store to root app
    if not only_sub_image_is_being_built(adr_map, app_output_dir):
        write_pm_config_to_file(path.join(app_output_dir, pm_kconfig_file), config_lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description='''Parse given 'pm.yml' partition manager configuration files to deduce the placement of partitions.

The partitions and their relative placement is defined in the 'pm.yml' files. The path to the 'pm.yml' files are used
to locate 'autoconf.h' files. These are used to find the partition sizes, as well as the total flash size.

This script generates a file for each partition - "pm_config.h".
This file contains all addresses and sizes of all partitions.

"pm_config.h" is in the same folder as the given 'pm.yml' file.''',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", required=True, type=str, nargs="+",
                        help="Space separated list of configs."
                             "Each config is a ':' - separated list of the following properties:"
                             "image-name:pm.yml-path:build_dir:out_dir")
    parser.add_argument("--app-pm-config-dir", required=True,
                        help="Where to store the 'pm_config.h' of the root app.")

    args = parser.parse_args()

    return args


def main():
    print("Running Partition Manager...")
    if len(sys.argv) > 1:
        args = parse_args()
        input_config = dict()
        for i in args.input:
            split = i.split('|')
            input_config[split[0]] = dict()
            input_config[split[0]]['pm.yml'] = split[1]
            input_config[split[0]]['build_dir'] = split[2]
            input_config[split[0]]['out_dir'] = split[3]
        pm_config = get_pm_config(input_config)
        write_pm_config(pm_config, args.app_pm_config_dir)
        write_kconfig_file(pm_config, args.app_pm_config_dir)
    else:
        print("No input, running tests.")
        test()

from pprint import pformat

def expect_addr_size(td, name, expected_address, expected_size):
    if expected_size:
        assert td[name]['size'] == expected_size, "Size of %s was %d, expected %d.\ntd:%s" % \
                                                  (name, td[name]['size'], expected_size, pformat(td))
    if expected_address:
        assert td[name]['address'] == expected_address, "Address of %s was %d, expected %d.\ntd:%s" % \
                                                        (name, td[name]['address'], expected_address, pformat(td))


def test():
    td = {'spm': {'placement': {'before': ['app']}, 'size': 100},
          'mcuboot': {'placement': {'before': ['spm', 'app']}, 'size': 200},
          'mcuboot_partitions': {'span': ['spm', 'app'], 'sub_partitions': ['primary', 'secondary']},
          'app': {'placement': ''}}
    s, sub_partitions = resolve(td)
    set_addresses(td, sub_partitions, s, 1000)
    set_sub_partition_address_and_size(td, sub_partitions)
    expect_addr_size(td, 'mcuboot', 0, None)
    expect_addr_size(td, 'spm', 200, None)
    expect_addr_size(td, 'app', 300, 700)
    expect_addr_size(td, 'mcuboot_partitions_primary', 200, 400)
    expect_addr_size(td, 'mcuboot_partitions_secondary', 600, 400)

    td = {'mcuboot': {'placement': {'before': ['app']}, 'size': 200},
          'mcuboot_partitions': {'span': ['app'], 'sub_partitions': ['primary', 'secondary']},
          'app': {'placement': ''}}
    s, sub_partitions = resolve(td)
    set_addresses(td, sub_partitions, s, 1000)
    set_sub_partition_address_and_size(td, sub_partitions)
    expect_addr_size(td, 'mcuboot', 0, None)
    expect_addr_size(td, 'app', 200, 800)
    expect_addr_size(td, 'mcuboot_partitions_primary', 200, 400)
    expect_addr_size(td, 'mcuboot_partitions_secondary', 600, 400)

    td = {'spm': {'placement': {'before': ['app']}, 'size': 100, 'inside': ['mcuboot_slot0']},
          'mcuboot': {'placement': {'before': ['app']}, 'size': 200},
          'mcuboot_pad': {'placement': {'after': ['mcuboot']}, 'inside': ['mcuboot_slot0'], 'size': 10},
          'app_partition': {'span': ['spm', 'app'], 'inside': ['mcuboot_slot0']},
          'mcuboot_slot0': {'span': ['app']},
          'mcuboot_data': {'placement': {'after': ['mcuboot_slot0']}, 'size': 200},
          'mcuboot_slot1': {'share_size': ['mcuboot_slot0'], 'placement': {'after': ['mcuboot_data']}},
          'mcuboot_slot2': {'share_size': ['mcuboot_slot1'], 'placement': {'after': ['mcuboot_slot1']}},
          'app': {'placement': ''}}
    s, sub_partitions = resolve(td)
    set_addresses(td, sub_partitions, s, 1000)
    set_sub_partition_address_and_size(td, sub_partitions)
    expect_addr_size(td, 'mcuboot', 0, None)
    expect_addr_size(td, 'spm', 210, None)
    expect_addr_size(td, 'mcuboot_slot0', 200, 200)
    expect_addr_size(td, 'mcuboot_slot1', 600, 200)
    expect_addr_size(td, 'mcuboot_slot2', 800, 200)
    expect_addr_size(td, 'app', 310, 90)
    expect_addr_size(td, 'mcuboot_pad', 200, 10)
    expect_addr_size(td, 'mcuboot_data', 400, 200)

    td = {
        'e': {'placement': {'before': ['app']}, 'size': 100},
        'a': {'placement': {'before': ['b']}, 'size': 100},
        'd': {'placement': {'before': ['e']}, 'size': 100},
        'c': {'placement': {'before': ['d']}, 'share_size': ['z', 'a', 'g']},
        'j': {'placement': 'last', 'size': 20},
        'i': {'placement': {'before': ['j']}, 'size': 20},
        'h': {'placement': {'before': ['i']}, 'size': 20},
        'f': {'placement': {'after': ['app']}, 'size': 20},
        'g': {'placement': {'after': ['f']}, 'size': 20},
        'b': {'placement': {'before': ['c']}, 'size': 20},
        'app': {'placement': ''}}
    s, _ = resolve(td)
    set_addresses(td, {}, s, 1000)
    expect_addr_size(td, 'a', 0, None)
    expect_addr_size(td, 'b', 100, None)
    expect_addr_size(td, 'c', 120, None)
    expect_addr_size(td, 'd', 220, None)
    expect_addr_size(td, 'e', 320, None)
    expect_addr_size(td, 'app', 420, 480)
    expect_addr_size(td, 'f', 900, None)
    expect_addr_size(td, 'g', 920, None)
    expect_addr_size(td, 'h', 940, None)
    expect_addr_size(td, 'i', 960, None)
    expect_addr_size(td, 'j', 980, None)

    td = {'mcuboot': {'placement': {'before': ['app', 'spu']}, 'size': 200},
          'b0': {'placement': {'before': ['mcuboot', 'app']}, 'size': 100},
          'app': {'placement': ''}}
    s, _ = resolve(td)
    set_addresses(td, {}, s, 1000)
    expect_addr_size(td, 'b0', 0, None)
    expect_addr_size(td, 'mcuboot', 100, None)
    expect_addr_size(td, 'app', 300, 700)

    td = {'b0': {'placement': {'before': ['mcuboot', 'app']}, 'size': 100}, 'app': {'placement': ''}}
    s, _ = resolve(td)
    set_addresses(td, {}, s, 1000)
    expect_addr_size(td, 'b0', 0, None)
    expect_addr_size(td, 'app', 100, 900)

    td = {'spu': {'placement': {'before': ['app']}, 'size': 100},
          'mcuboot': {'placement': {'before': ['spu', 'app']}, 'size': 200},
          'app': {'placement': ''}}
    s, _ = resolve(td)
    set_addresses(td, {}, s, 1000)
    expect_addr_size(td, 'mcuboot', 0, None)
    expect_addr_size(td, 'spu', 200, None)
    expect_addr_size(td, 'app', 300, 700)

    td = {'provision': {'placement': 'last', 'size': 100},
          'mcuboot': {'placement': {'before': ['spu', 'app']}, 'size': 100},
          'b0': {'placement': {'before': ['mcuboot', 'app']}, 'size': 50},
          'spu': {'placement': {'before': ['app']}, 'size': 100},
          'app': {'placement': ''}}
    s, _ = resolve(td)
    set_addresses(td, {}, s, 1000)
    expect_addr_size(td, 'b0', 0, None)
    expect_addr_size(td, 'mcuboot', 50, None)
    expect_addr_size(td, 'spu', 150, None)
    expect_addr_size(td, 'app', 250, 650)
    expect_addr_size(td, 'provision', 900, None)

    print("All tests passed!")


if __name__ == "__main__":
    main()
