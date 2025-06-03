import { Tabs } from 'expo-router'
import { EvilIcons } from '@expo/vector-icons';
import TabHeader from '../../_components/TabHeader';

const ICON_SIZE = 30;

interface optionsInterface {
    title: string,
    tintColor: string,
    iconName: "comment" | "image" | "gear",
    addButton: boolean
}

const options = ({ title, tintColor, iconName, addButton }: optionsInterface) => {
    return {
        header: () => <TabHeader title={title} addButton={addButton} />,
        tabBarActiveTintColor: tintColor,
        tabBarLabel: title,
        tabBarIcon: (({ color }: any) => <EvilIcons color={color} name={iconName} size={ICON_SIZE} />),
    }
}

export default function _layout() {
    return <Tabs screenOptions={
        {
            animation: 'shift',
            headerTitleAlign: 'left',
            tabBarStyle: { height: 60 },
            tabBarLabelPosition: 'below-icon',
        }
    }>
        <Tabs.Screen name='index' options={options({ title: 'chats', tintColor: 'green', iconName: 'comment', addButton: true })} />

        <Tabs.Screen name='explore' options={options({ title: 'explore', tintColor: 'red', iconName: 'image', addButton: false })} />

        <Tabs.Screen name='settings' options={options({ title: 'settings', tintColor: 'yellow', iconName: 'gear', addButton: false })} />
    </Tabs>;
}


